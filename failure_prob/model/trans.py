import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.use_prefix_pairwise_auc = cfg.model.use_prefix_pairwise_auc
        self.lambda_prefix_pairwise_auc = cfg.model.lambda_prefix_pairwise_auc
        self.prefix_pairwise_ratio = cfg.model.prefix_pairwise_ratio
        self.prefix_pairwise_ratios = self._resolve_prefix_ratios(
            getattr(cfg.model, "prefix_pairwise_ratios", None),
            fallback_ratio=self.prefix_pairwise_ratio,
        )
        self.prefix_pairwise_weights = self._resolve_ratio_weights(
            getattr(cfg.model, "prefix_pairwise_weights", None),
            self.prefix_pairwise_ratios,
            "prefix_pairwise_weights",
        )
        self.prefix_pairwise_time_discount_gamma = float(
            getattr(cfg.model, "prefix_pairwise_time_discount_gamma", 0.0)
        )
        if self.prefix_pairwise_time_discount_gamma < 0.0:
            raise ValueError(
                "prefix_pairwise_time_discount_gamma must be non-negative"
            )
        self.use_integral_pairwise_loss = bool(
            getattr(cfg.model, "use_integral_pairwise_loss", False)
        )
        self.lambda_integral_pairwise_loss = float(
            getattr(cfg.model, "lambda_integral_pairwise_loss", 0.0)
        )
        self.integral_pairwise_gamma = float(
            getattr(cfg.model, "integral_pairwise_gamma", 0.0)
        )
        if self.integral_pairwise_gamma < 0.0:
            raise ValueError("integral_pairwise_gamma must be non-negative")
        self.lambda_prefix_monitor = cfg.model.lambda_prefix_monitor
        self.prefix_monitor_ratio = cfg.model.prefix_monitor_ratio
        self.prefix_monitor_ratios = self._resolve_prefix_ratios(
            getattr(cfg.model, "prefix_monitor_ratios", None),
            fallback_ratio=self.prefix_monitor_ratio,
        )
        self.prefix_monitor_weights = self._resolve_ratio_weights(
            getattr(cfg.model, "prefix_monitor_weights", None),
            self.prefix_monitor_ratios,
            "prefix_monitor_weights",
        )
        self.use_class_conditional_time_weights = cfg.model.use_class_conditional_time_weights
        self.use_soft_detection_loss = cfg.model.use_soft_detection_loss
        self.lambda_soft_detection = cfg.model.lambda_soft_detection
        self.soft_detection_threshold = cfg.model.soft_detection_threshold
        self.soft_detection_temperature = cfg.model.soft_detection_temperature
        if self.use_time_gate:
            # Learnable gate center in normalized time (0..1)
            eps = 1e-4
            init_tau = float(cfg.model.time_gate_tau_init)
            init_tau = min(max(init_tau, eps), 1.0 - eps)
            self.time_gate_tau = nn.Parameter(torch.logit(torch.tensor(init_tau)))
        else:
            self.time_gate_tau = None

        self.aux_warmup_epochs = max(int(cfg.model.aux_warmup_epochs), 0)
        self.aux_ramp_epochs = max(int(cfg.model.aux_ramp_epochs), 0)
        self._train_epoch_idx = 0
        
        self._scale_weights(self.cfg.model.init_weight_scale)


    def _resolve_prefix_ratios(
        self,
        ratios: list[float] | tuple[float, ...] | None,
        fallback_ratio: float,
    ) -> list[float]:
        if ratios is None:
            ratios = []

        resolved = []
        for ratio in list(ratios):
            ratio = float(ratio)
            if not (0.0 < ratio <= 1.0):
                raise ValueError(f"prefix ratio must be in (0, 1], got {ratio}")
            if any(math.isclose(ratio, existing) for existing in resolved):
                continue
            resolved.append(ratio)

        if not resolved:
            fallback_ratio = float(fallback_ratio)
            if not (0.0 < fallback_ratio <= 1.0):
                raise ValueError(f"prefix ratio must be in (0, 1], got {fallback_ratio}")
            resolved.append(fallback_ratio)

        return resolved


    def _resolve_ratio_weights(
        self,
        weights: list[float] | tuple[float, ...] | None,
        ratios: list[float],
        name: str,
    ) -> list[float]:
        if weights is None:
            weights = []

        if len(weights) == 0:
            return [1.0] * len(ratios)
        if len(weights) != len(ratios):
            raise ValueError(
                f"{name} length must match ratios length: {len(weights)} != {len(ratios)}"
            )

        resolved = []
        for weight in list(weights):
            weight = float(weight)
            if weight <= 0.0:
                raise ValueError(f"{name} entries must be positive, got {weight}")
            resolved.append(weight)
        return resolved


    def _weighted_mean(
        self,
        losses: list[torch.Tensor],
        weights: list[float],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if not losses:
            return ref.new_tensor(0.0)
        loss_tensor = torch.stack(losses)
        weight_tensor = ref.new_tensor(weights)
        return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-12)


    def _build_normalized_time_index(self, valid_masks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        valid_masks = valid_masks.to(dtype=dtype)
        B, T = valid_masks.shape
        seq_lengths = valid_masks.sum(dim=1).clamp(min=1.0)
        denom = torch.clamp(seq_lengths - 1.0, min=1.0)
        t_idx = torch.arange(T, device=valid_masks.device, dtype=dtype).unsqueeze(0).expand(B, -1)
        t_norm = torch.minimum(t_idx / denom.unsqueeze(1), torch.ones_like(t_idx))
        return t_norm * valid_masks


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
            # For each timestep t, use the most recent n steps including the current one.
            pad_steps = max(n - 1, 0)
            x_padded = torch.nn.functional.pad(x, (0, 0, pad_steps, 0), mode="constant", value=0)

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
        time_discount_gamma: float = 0.0,
    ) -> torch.Tensor:
        # Compute a differentiable pairwise AUC loss on sequence-level scores.
        # scores: (B, T), labels: (B,), valid_masks: (B, T)
        # labels should be failure labels: 1 for failure (positive), 0 for success (negative).
        if scores.numel() == 0:
            return scores.new_tensor(0.0)

        adjusted_scores = scores
        if time_discount_gamma > 0.0:
            t_norm = self._build_normalized_time_index(valid_masks, scores.dtype)
            adjusted_scores = adjusted_scores - float(time_discount_gamma) * t_norm

        masked_scores = torch.where(
            valid_masks > 0.5,
            adjusted_scores,
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


    def _pairwise_ranking_loss(
        self,
        seq_scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        pos = seq_scores[labels > 0.5]
        neg = seq_scores[labels <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return seq_scores.new_tensor(0.0)

        diff = pos[:, None] - neg[None, :]
        return F.softplus(-diff).mean()


    def _integral_sequence_scores(
        self,
        scores: torch.Tensor,
        valid_masks: torch.Tensor,
        gamma: float = 0.0,
    ) -> torch.Tensor:
        valid_masks = valid_masks.to(dtype=scores.dtype)
        weights = valid_masks
        if gamma > 0.0:
            t_norm = self._build_normalized_time_index(valid_masks, scores.dtype)
            weights = weights * torch.exp(-float(gamma) * t_norm)

        denom = weights.sum(dim=1).clamp_min(1e-12)
        return (scores * weights).sum(dim=1) / denom


    def _integral_pairwise_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_masks: torch.Tensor,
        gamma: float = 0.0,
    ) -> torch.Tensor:
        seq_scores = self._integral_sequence_scores(
            scores,
            valid_masks,
            gamma=gamma,
        )
        return self._pairwise_ranking_loss(seq_scores, labels)


    def _build_prefix_valid_masks(
        self,
        valid_masks: torch.Tensor,
        ratio: float | None = None,
    ) -> torch.Tensor:
        # Build a prefix-only valid mask per sequence to approximate early-window optimization.
        ratio = float(self.prefix_pairwise_ratio if ratio is None else ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"prefix ratio must be in (0, 1], got {ratio}"
            )

        B, T = valid_masks.shape
        seq_lengths = valid_masks.sum(dim=1).long().clamp(min=1, max=T)  # (B,)
        prefix_lengths = torch.ceil(seq_lengths.float() * ratio).long()
        prefix_lengths = prefix_lengths.clamp(min=1, max=T)  # (B,)

        t_idx = torch.arange(T, device=valid_masks.device).unsqueeze(0).expand(B, -1)  # (B, T)
        prefix_masks = (t_idx < prefix_lengths.unsqueeze(1)).to(valid_masks.dtype)  # (B, T)
        return prefix_masks * valid_masks


    def _get_aux_loss_scale(self) -> float:
        epoch_idx = max(int(self._train_epoch_idx), 0)
        if epoch_idx <= self.aux_warmup_epochs:
            return 0.0
        if self.aux_ramp_epochs == 0:
            return 1.0

        progress = (epoch_idx - self.aux_warmup_epochs) / float(self.aux_ramp_epochs)
        return float(min(max(progress, 0.0), 1.0))


    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        dataloader,
    ) -> float:
        self._train_epoch_idx += 1
        return super().train_epoch(optimizer, dataloader)


    def _build_time_weights(
        self,
        valid_masks: torch.Tensor,
        success_labels: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        failure_time_weights = get_time_weight(self.cfg.model.use_time_weighting, valid_masks).to(dtype=dtype)
        if not self.use_class_conditional_time_weights:
            return failure_time_weights

        # Keep success weighting uniform so the model is not pushed to suppress all early scores.
        success_time_weights = valid_masks.to(dtype=dtype)
        return torch.where(success_labels[:, None] > 0.5, success_time_weights, failure_time_weights)


    def _soft_detection_loss(
        self,
        scores: torch.Tensor,
        failure_labels: torch.Tensor,
        valid_masks: torch.Tensor,
    ) -> torch.Tensor:
        if scores.numel() == 0:
            return scores.new_tensor(0.0)

        eps = 1e-6
        threshold = float(self.soft_detection_threshold)
        temperature = max(float(self.soft_detection_temperature), eps)
        valid_masks = valid_masks.to(scores)

        detect_probs = torch.sigmoid((scores - threshold) / temperature) * valid_masks
        detect_probs = detect_probs.clamp(min=0.0, max=1.0 - eps)

        t_norm = self._build_normalized_time_index(valid_masks, scores.dtype)

        log_survival = torch.cumsum(torch.log(torch.clamp(1.0 - detect_probs, min=eps)), dim=1)
        prev_log_survival = F.pad(log_survival[:, :-1], (1, 0), value=0.0)
        prev_survival = torch.exp(prev_log_survival)
        hit_mass = prev_survival * detect_probs
        final_survival = torch.exp(log_survival[:, -1])

        expected_det_time = (hit_mass * t_norm).sum(dim=1) + final_survival
        success_alarm = detect_probs.max(dim=1).values

        loss_terms = []
        pos_mask = failure_labels > 0.5
        neg_mask = ~pos_mask
        if pos_mask.any():
            loss_terms.append(expected_det_time[pos_mask].mean())
        if neg_mask.any():
            loss_terms.append(success_alarm[neg_mask].mean())
        if not loss_terms:
            return scores.new_tensor(0.0)
        return torch.stack(loss_terms).mean()


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
        time_weights = self._build_time_weights(valid_masks, success_labels, scores.dtype).to(scores)
        
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
        aux_loss_scale = self._get_aux_loss_scale()

        prefix_monitor_loss = torch.tensor(0.0).to(scores)
        if self.lambda_prefix_monitor > 0:
            prefix_monitor_losses = []
            prefix_monitor_weights = []
            for ratio, ratio_weight in zip(self.prefix_monitor_ratios, self.prefix_monitor_weights):
                prefix_valid_masks = self._build_prefix_valid_masks(valid_masks, ratio=ratio)
                ratio_loss, _, _ = aggregate_monitor_loss(
                    losses,
                    prefix_valid_masks,
                    success_labels,
                    weights,
                    self.cfg.model.one_loss_per_seq,
                )
                prefix_monitor_losses.append(ratio_loss)
                prefix_monitor_weights.append(ratio_weight)
            prefix_monitor_loss = self._weighted_mean(
                prefix_monitor_losses,
                prefix_monitor_weights,
                scores,
            )
            prefix_monitor_loss = aux_loss_scale * self.lambda_prefix_monitor * prefix_monitor_loss

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
        
        # Align pairwise AUC with BCE target: failure is positive class.
        failure_labels = 1 - success_labels

        pairwise_auc_loss = torch.tensor(0.0).to(scores)
        if self.use_pairwise_auc and self.lambda_pairwise_auc > 0:
            pairwise_auc_loss = self._pairwise_auc_loss(scores, failure_labels, valid_masks)
            pairwise_auc_loss = aux_loss_scale * self.lambda_pairwise_auc * pairwise_auc_loss

        prefix_pairwise_auc_loss = torch.tensor(0.0).to(scores)
        if self.use_prefix_pairwise_auc and self.lambda_prefix_pairwise_auc > 0:
            prefix_pairwise_losses = []
            prefix_pairwise_weights = []
            for ratio, ratio_weight in zip(self.prefix_pairwise_ratios, self.prefix_pairwise_weights):
                prefix_valid_masks = self._build_prefix_valid_masks(valid_masks, ratio=ratio)
                ratio_loss = self._pairwise_auc_loss(
                    scores,
                    failure_labels,
                    prefix_valid_masks,
                    time_discount_gamma=self.prefix_pairwise_time_discount_gamma,
                )
                prefix_pairwise_losses.append(ratio_loss)
                prefix_pairwise_weights.append(ratio_weight)
            prefix_pairwise_auc_loss = self._weighted_mean(
                prefix_pairwise_losses,
                prefix_pairwise_weights,
                scores,
            )
            prefix_pairwise_auc_loss = aux_loss_scale * self.lambda_prefix_pairwise_auc * prefix_pairwise_auc_loss

        integral_pairwise_loss = torch.tensor(0.0).to(scores)
        if self.use_integral_pairwise_loss and self.lambda_integral_pairwise_loss > 0:
            integral_pairwise_loss = self._integral_pairwise_loss(
                scores,
                failure_labels,
                valid_masks,
                gamma=self.integral_pairwise_gamma,
            )
            integral_pairwise_loss = (
                aux_loss_scale * self.lambda_integral_pairwise_loss * integral_pairwise_loss
            )

        soft_detection_loss = torch.tensor(0.0).to(scores)
        if self.use_soft_detection_loss and self.lambda_soft_detection > 0:
            soft_detection_loss = self._soft_detection_loss(scores, failure_labels, valid_masks)
            soft_detection_loss = aux_loss_scale * self.lambda_soft_detection * soft_detection_loss

        monitor_loss += (
            prefix_monitor_loss
            + hard_neg_loss
            + pairwise_auc_loss
            + prefix_pairwise_auc_loss
            + integral_pairwise_loss
            + soft_detection_loss
        )

        # Log the losses
        logs = {
            "monitor_loss": monitor_loss.item(),
            "success_loss": success_loss.item(),
            "fail_loss": fail_loss.item(),
            "aux_loss_scale": float(aux_loss_scale),
            "prefix_monitor_loss": prefix_monitor_loss.item(),
            "hard_neg_loss": hard_neg_loss.item(),
            "pairwise_auc_loss": pairwise_auc_loss.item(),
            "prefix_pairwise_auc_loss": prefix_pairwise_auc_loss.item(),
            "integral_pairwise_loss": integral_pairwise_loss.item(),
            "soft_detection_loss": soft_detection_loss.item(),
        }
        
        return monitor_loss, logs
