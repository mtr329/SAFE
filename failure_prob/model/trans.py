import math
import pdb
import torch
import torch.nn as nn

from .base import BaseModel
from .utils import get_time_weight, aggregate_monitor_loss, hard_negative_loss, cumsum_stopgrad

from failure_prob.conf import Config


def get_model(cfg, input_dim):
    return TransModel(cfg, input_dim)


class TransModel(BaseModel):
    def __init__(self, cfg: Config, input_dim: int):
        super().__init__(cfg, input_dim)
        self.hidden_dim = cfg.model.hidden_dim
        self.n_layers = cfg.model.n_layers
        self.n_heads = cfg.model.n_heads
        self.ff_dim = cfg.model.ff_dim
        assert self.hidden_dim % self.n_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
        )

        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.ff_dim,
            dropout=cfg.model.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.n_history_steps = cfg.model.n_history_steps
        self.use_time_gate = cfg.model.use_time_gate
        self.time_gate_k = cfg.model.time_gate_k
        self.time_gate_a = cfg.model.time_gate_a
        self.use_pairwise_auc = cfg.model.use_pairwise_auc
        self.lambda_pairwise_auc = cfg.model.lambda_pairwise_auc
        self.pairwise_auc_beta = cfg.model.pairwise_auc_beta
        if self.use_time_gate:
            # Learnable gate center in normalized time (0..1)
            eps = 1e-4
            init_tau = float(cfg.model.time_gate_tau_init)
            init_tau = min(max(init_tau, eps), 1.0 - eps)
            self.time_gate_tau = nn.Parameter(torch.logit(torch.tensor(init_tau)))
        else:
            self.time_gate_tau = None
        
        self._scale_weights(self.cfg.model.init_weight_scale)


    def _positional_encoding(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.hidden_dim)
        )
        pe = torch.zeros(length, self.hidden_dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(positions * div_term)
        if self.hidden_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(positions * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)  # (1, T, D)


    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
    

    def forward(
        self, 
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = batch["features"]
        B, T, D = x.shape
        n = self.n_history_steps

        assert x.ndim == 3, f"Input dim mismatch: {x.ndim} != 3"
        assert D == self.input_dim, f"Input dim mismatch: {D} != {self.input_dim}"

        if n < 0:
            x_proj = self.input_proj(x)  # (B, T, hidden_dim)
            x_proj = x_proj + self._positional_encoding(T, x.device, x_proj.dtype)
            x_proj = self.dropout(x_proj)
            out = self.encoder(x_proj, mask=self._causal_mask(T, x.device))  # (B, T, hidden_dim)
        else:
            # Prepare sliding windows: for each timestep t, extract [t-n, ..., t-1]
            x_padded = torch.nn.functional.pad(x, (0, 0, n, 0), mode="constant", value=0)  # (B, T+n, D)

            x_windows = []
            for t in range(T):
                x_window = x_padded[:, t:t+n, :]  # (B, n, D)
                x_windows.append(x_window)

            x_seq = torch.stack(x_windows, dim=1)  # (B, T, n, D)
            x_seq = x_seq.reshape(B * T, n, D)     # (B*T, n, D)

            x_proj = self.input_proj(x_seq)        # (B*T, n, hidden_dim)
            x_proj = x_proj + self._positional_encoding(n, x.device, x_proj.dtype)
            x_proj = self.dropout(x_proj)
            out = self.encoder(x_proj, mask=self._causal_mask(n, x.device))  # (B*T, n, hidden_dim)
            out = out[:, -1, :]                    # (B*T, hidden_dim)
            out = out.view(B, T, -1)               # (B, T, hidden_dim)

        out = self.dropout(out)                 # (B, T, hidden_dim)
        p_seq = torch.sigmoid(self.fc(out))    # (B, T, 1)

        if self.cfg.model.cumsum:
            # p_seq = p_seq.cumsum(dim=1)
            p_seq = cumsum_stopgrad(p_seq, dim=1)  # (B, T, 1)
            if self.cfg.model.rmean:
                normalizer = p_seq.new_ones(p_seq.shape).cumsum(dim=1)
                p_seq = p_seq / normalizer

        return p_seq


    def _pairwise_auc_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_masks: torch.Tensor,
    ) -> torch.Tensor:
        # Compute a differentiable pairwise AUC loss on sequence-level scores.
        # scores: (B, T), labels: (B,), valid_masks: (B, T)
        if scores.numel() == 0:
            return scores.new_tensor(0.0)

        masked_scores = torch.where(
            valid_masks > 0.5,
            scores,
            torch.tensor(-float("inf"), device=scores.device, dtype=scores.dtype),
        )
        beta = float(self.pairwise_auc_beta)
        # Soft maximum over time for each sequence.
        s = (1.0 / beta) * torch.logsumexp(beta * masked_scores, dim=1)  # (B,)

        pos = s[labels == 1]
        neg = s[labels == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return scores.new_tensor(0.0)

        # Pairwise logistic loss: encourage pos > neg.
        diff = pos[:, None] - neg[None, :]
        return torch.nn.functional.softplus(-diff).mean()


    def forward_compute_loss(
        self, 
        batch: dict[str, torch.Tensor],
        weights: list[float] = None, 
    ) -> tuple[torch.Tensor, dict[str, float]]:
        valid_masks = batch["valid_masks"]
        success_labels = batch["success_labels"]
        B, T, D = batch["features"].shape

        # Break into pdb if inputs contain NaN/Inf before computing scores.
        if not torch.isfinite(batch["features"]).all():
            pdb.set_trace()
        if not torch.isfinite(valid_masks).all():
            pdb.set_trace()
        if not torch.isfinite(success_labels).all():
            pdb.set_trace()
        
        scores = self(batch)  # (B, T, 1)
        scores = scores.squeeze(-1)  # (B, T)
        
        # Design the weights based on time
        time_weights = get_time_weight(self.cfg.model.use_time_weighting, valid_masks)  # (B, T)
        time_weights = time_weights.to(scores) # (B, T)
        
        if self.use_time_gate:
            # Build a learnable time gate g(t) in [0,1] to control when class separation increases.
            # Early: g~0 (pos/neg weights similar). Late: g~1 (pos/neg weights diverge).
            t_idx = torch.arange(T, device=scores.device, dtype=scores.dtype).unsqueeze(0)  # (1, T)
            # Absolute time alignment: normalize by the global sequence length (T)
            denom = max(T - 1, 1)
            t_norm = t_idx / float(denom)  # (1, T)
            tau = torch.sigmoid(self.time_gate_tau)  # scalar in (0,1)
            gate = torch.sigmoid(self.time_gate_k * (t_norm - tau)) * valid_masks  # (B, T)
            pos_gate_w = 1.0 + self.time_gate_a * gate  # (B, T)
            neg_gate_w = 1.0 - self.time_gate_a * gate  # (B, T)
        else:
            pos_gate_w = None
            neg_gate_w = None
        
        if self.cfg.model.cumsum:
            # Compute the loss as if each sequence is successful or failure, then aggregate back to (B, T)
            lower_thresh = 0
            seq_loss_success = torch.relu(scores - lower_thresh)  # (B, T)
            seq_loss_success = time_weights * seq_loss_success
            if self.use_time_gate:
                seq_loss_success = pos_gate_w * seq_loss_success
            seq_loss_fail = time_weights * (- scores)
            if self.use_time_gate:
                seq_loss_fail = neg_gate_w * seq_loss_fail
                
            losses = (success_labels == 1).float()[:, None] * seq_loss_success + \
                (success_labels == 0).float()[:, None] * seq_loss_fail  # (B, T)
        else:
            # Compute BCE loss on scores at all timesteps
            criterion = nn.BCELoss(reduction="none")
            # Failure is the positive class
            if scores.isnan().any():
                raise RuntimeError("NaN detected in scores")

            losses = criterion(scores, 1 - success_labels.unsqueeze(-1).expand_as(scores)) # (B, T)
            
            if self.use_time_gate:
                losses[success_labels == 1] *= pos_gate_w[success_labels == 1]
                losses[success_labels == 0] *= neg_gate_w[success_labels == 0]
            
            # Apply time weights on both success and failure samples
            losses = losses * time_weights
        
        monitor_loss, success_loss, fail_loss = aggregate_monitor_loss(
            losses, valid_masks, success_labels, weights,
            self.cfg.model.one_loss_per_seq,
        )

        # Now that we want to do classification based on the max scores before termination
        # Therefore add hard nagative mining loss
        hard_neg_loss = torch.tensor(0.0).to(scores)
        if self.cfg.model.lambda_hard_heg > 0:
            # Note that in success_labels==0 means failure
            hard_neg_loss = hard_negative_loss(
                scores, 1-success_labels, valid_masks, 
                self.cfg.model.hard_neg_margin, 
                self.cfg.model.hard_neg_beta
            )
            hard_neg_loss = self.cfg.model.lambda_hard_heg * hard_neg_loss
        
        pairwise_auc_loss = torch.tensor(0.0).to(scores)
        if self.use_pairwise_auc and self.lambda_pairwise_auc > 0:
            pairwise_auc_loss = self._pairwise_auc_loss(scores, success_labels, valid_masks)
            pairwise_auc_loss = self.lambda_pairwise_auc * pairwise_auc_loss
        
        monitor_loss += hard_neg_loss + pairwise_auc_loss

        # Log the losses
        logs = {
            "monitor_loss": monitor_loss.item(),
            "success_loss": success_loss.item(),
            "fail_loss": fail_loss.item(),
            "hard_neg_loss": hard_neg_loss.item(),
            "pairwise_auc_loss": pairwise_auc_loss.item(),
        }
        
        return monitor_loss, logs
