from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.features.dataset import get_channel_counts


class ConvStem(nn.Module):
    def __init__(self, in_ch: int, hidden: int, latent_dim: int, dropout: float):
        super().__init__()
        mid = max(hidden, latent_dim // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(mid, latent_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPooling(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        hidden = max(32, latent_dim // 2)
        self.score = nn.Sequential(
            nn.Conv1d(latent_dim, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(x).squeeze(1)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(1), dim=-1)
        return pooled, weights


class ModalityAwareEncoder(nn.Module):
    def __init__(
        self,
        modality: str,
        eeg_in_ch: int,
        emg_in_ch: int,
        stem_hidden: int = 64,
        latent_dim: int = 96,
        pooling: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modality = modality
        self.eeg_in_ch = eeg_in_ch
        self.emg_in_ch = emg_in_ch
        self.pooling = pooling

        if modality in ("eeg", "fusion"):
            self.eeg_stem = ConvStem(eeg_in_ch, stem_hidden, latent_dim, dropout)
        else:
            self.eeg_stem = None

        if modality in ("emg", "fusion"):
            self.emg_stem = ConvStem(emg_in_ch, stem_hidden, latent_dim, dropout)
        else:
            self.emg_stem = None

        if modality == "fusion":
            gate_hidden = max(32, latent_dim // 2)
            self.fusion_gate = nn.Sequential(
                nn.Conv1d(latent_dim * 3, gate_hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(gate_hidden, latent_dim, kernel_size=1),
                nn.Sigmoid(),
            )
            self.shared_mix = nn.Conv1d(latent_dim * 2, latent_dim, kernel_size=1)
        else:
            self.fusion_gate = None
            self.shared_mix = None

        self.pool = AttentionPooling(latent_dim) if pooling == "attention" else nn.AdaptiveAvgPool1d(1)

    def _split_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.modality == "fusion":
            eeg = x[:, :self.eeg_in_ch, :]
            emg = x[:, self.eeg_in_ch:self.eeg_in_ch + self.emg_in_ch, :]
            return eeg, emg
        if self.modality == "eeg":
            return x, None
        return None, x

    def _pool(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(self.pool, AttentionPooling):
            return self.pool(x)
        pooled = self.pool(x).squeeze(-1)
        return pooled, None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eeg_x, emg_x = self._split_inputs(x)
        aux: dict[str, torch.Tensor] = {}

        if self.modality == "eeg":
            seq = self.eeg_stem(eeg_x)
        elif self.modality == "emg":
            seq = self.emg_stem(emg_x)
        else:
            eeg_seq = self.eeg_stem(eeg_x)
            emg_seq = self.emg_stem(emg_x)
            gate_input = torch.cat([eeg_seq, emg_seq, eeg_seq - emg_seq], dim=1)
            gate = self.fusion_gate(gate_input)
            mix = self.shared_mix(torch.cat([eeg_seq, emg_seq], dim=1))
            seq = gate * emg_seq + (1.0 - gate) * eeg_seq + 0.5 * mix
            aux["fusion_gate_mean"] = gate.mean(dim=(1, 2))

        pooled, weights = self._pool(seq)
        if weights is not None:
            aux["attn_mean"] = weights.mean(dim=1)
        return pooled, aux


class PhaseClassifier(nn.Module):
    def __init__(
        self,
        modality: str,
        eeg_in_ch: int,
        emg_in_ch: int,
        stem_hidden: int = 64,
        latent_dim: int = 96,
        head_hidden: int = 64,
        pooling: str = "attention",
        dropout: float = 0.1,
        n_classes: int = 4,
    ):
        super().__init__()
        self.encoder = ModalityAwareEncoder(
            modality=modality,
            eeg_in_ch=eeg_in_ch,
            emg_in_ch=emg_in_ch,
            stem_hidden=stem_hidden,
            latent_dim=latent_dim,
            pooling=pooling,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x)
        return self.head(h)


class TorqueRegressor(nn.Module):
    def __init__(
        self,
        modality: str,
        eeg_in_ch: int,
        emg_in_ch: int,
        stem_hidden: int = 64,
        latent_dim: int = 96,
        head_hidden: int = 64,
        pooling: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ModalityAwareEncoder(
            modality=modality,
            eeg_in_ch=eeg_in_ch,
            emg_in_ch=emg_in_ch,
            stem_hidden=stem_hidden,
            latent_dim=latent_dim,
            pooling=pooling,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x)
        return self.head(h).squeeze(-1)


class LegacyConvBackbone(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return self.pool(h).squeeze(-1)


class LegacyPhaseClassifier(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 4):
        super().__init__()
        self.backbone = LegacyConvBackbone(in_ch, hidden=96)
        self.head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class LegacyTorqueRegressor(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.backbone = LegacyConvBackbone(in_ch, hidden=96)
        self.head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


def _task_model_cfg(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    return cfg.get("model", {}).get(task, {})


def get_model_kwargs(cfg: dict[str, Any], task: str, modality: str) -> dict[str, Any]:
    eeg_in_ch, emg_in_ch = get_channel_counts(cfg)
    model_cfg = _task_model_cfg(cfg, task)
    kwargs = {
        "modality": modality,
        "eeg_in_ch": eeg_in_ch,
        "emg_in_ch": emg_in_ch,
        "stem_hidden": int(model_cfg.get("stem_hidden", 64)),
        "latent_dim": int(model_cfg.get("latent_dim", 96)),
        "head_hidden": int(model_cfg.get("head_hidden", 64)),
        "pooling": str(model_cfg.get("pooling", "attention")),
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }
    if task == "phase":
        kwargs["n_classes"] = int(model_cfg.get("n_classes", 4))
    return kwargs


def build_phase_model(cfg: dict[str, Any], modality: str) -> PhaseClassifier:
    return PhaseClassifier(**get_model_kwargs(cfg, "phase", modality))


def build_torque_model(cfg: dict[str, Any], modality: str) -> TorqueRegressor:
    return TorqueRegressor(**get_model_kwargs(cfg, "torque", modality))


def save_model_checkpoint(
    path: str | Any,
    model: nn.Module,
    cfg: dict[str, Any],
    task: str,
    modality: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "task": task,
        "modality": modality,
        "model_kwargs": get_model_kwargs(cfg, task, modality),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_model_checkpoint(
    path: str | Any,
    cfg: dict[str, Any],
    task: str,
    modality: str,
    device: torch.device,
) -> nn.Module:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload and "model_kwargs" in payload:
        model_kwargs = payload["model_kwargs"]
        model = PhaseClassifier(**model_kwargs) if task == "phase" else TorqueRegressor(**model_kwargs)
        model.load_state_dict(payload["state_dict"])
    else:
        eeg_in_ch, emg_in_ch = get_channel_counts(cfg)
        if modality == "fusion":
            in_ch = eeg_in_ch + emg_in_ch
        elif modality == "eeg":
            in_ch = eeg_in_ch
        else:
            in_ch = emg_in_ch
        model = LegacyPhaseClassifier(in_ch=in_ch) if task == "phase" else LegacyTorqueRegressor(in_ch=in_ch)
        model.load_state_dict(payload)
    return model.to(device)
