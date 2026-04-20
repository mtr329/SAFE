import math

import torch
import torch.nn as nn

from .base import BaseModel
from .utils import aggregate_monitor_loss, cumsum_stopgrad, get_time_weight

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
        self.use_class_conditional_time_weights = (
            cfg.model.use_class_conditional_time_weights
        )

        self._scale_weights(self.cfg.model.init_weight_scale)

    def _positional_encoding(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
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
        return pe.unsqueeze(0)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _build_time_weights(
        self,
        valid_masks: torch.Tensor,
        success_labels: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        failure_time_weights = get_time_weight(
            self.cfg.model.use_time_weighting,
            valid_masks,
        ).to(dtype=dtype)
        if not self.use_class_conditional_time_weights:
            return failure_time_weights

        success_time_weights = valid_masks.to(dtype=dtype)
        return torch.where(
            success_labels[:, None] > 0.5,
            success_time_weights,
            failure_time_weights,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = batch["features"]
        batch_size, seq_len, feat_dim = x.shape
        n = self.n_history_steps

        assert x.ndim == 3, f"Input dim mismatch: {x.ndim} != 3"
        assert feat_dim == self.input_dim, (
            f"Input dim mismatch: {feat_dim} != {self.input_dim}"
        )

        if n < 0:
            x_proj = self.input_proj(x)
            x_proj = x_proj + self._positional_encoding(seq_len, x.device, x_proj.dtype)
            x_proj = self.dropout(x_proj)
            out = self.encoder(x_proj, mask=self._causal_mask(seq_len, x.device))
        else:
            pad_steps = max(n - 1, 0)
            x_padded = torch.nn.functional.pad(
                x,
                (0, 0, pad_steps, 0),
                mode="constant",
                value=0,
            )

            x_windows = []
            for t in range(seq_len):
                x_windows.append(x_padded[:, t:t + n, :])

            x_seq = torch.stack(x_windows, dim=1)
            x_seq = x_seq.reshape(batch_size * seq_len, n, feat_dim)

            x_proj = self.input_proj(x_seq)
            x_proj = x_proj + self._positional_encoding(n, x.device, x_proj.dtype)
            x_proj = self.dropout(x_proj)
            out = self.encoder(x_proj, mask=self._causal_mask(n, x.device))
            out = out[:, -1, :]
            out = out.view(batch_size, seq_len, -1)

        out = self.dropout(out)
        p_seq = torch.sigmoid(self.fc(out))

        if self.cfg.model.cumsum:
            p_seq = cumsum_stopgrad(p_seq, dim=1)

        return p_seq

    def forward_compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        weights: list[float] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        valid_masks = batch["valid_masks"]
        success_labels = batch["success_labels"]

        scores = self(batch).squeeze(-1)
        time_weights = self._build_time_weights(
            valid_masks,
            success_labels,
            scores.dtype,
        ).to(scores)

        if self.cfg.model.cumsum:
            seq_loss_success = time_weights * torch.relu(scores)
            seq_loss_fail = time_weights * (-scores)
            losses = (
                (success_labels == 1).float()[:, None] * seq_loss_success
                + (success_labels == 0).float()[:, None] * seq_loss_fail
            )
        else:
            criterion = nn.BCELoss(reduction="none")
            targets = 1 - success_labels.unsqueeze(-1).expand_as(scores)
            losses = criterion(scores, targets) * time_weights

        if weights is None:
            weights = [1.0, 1.0]

        monitor_loss, success_loss, fail_loss = aggregate_monitor_loss(
            losses,
            valid_masks,
            success_labels,
            weights,
        )

        logs = {
            "monitor_loss": monitor_loss.item(),
            "success_loss": success_loss.item(),
            "fail_loss": fail_loss.item(),
        }
        return monitor_loss, logs
