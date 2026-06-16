from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints


@dataclass(frozen=True)
class ModelConfig:
    ehr_features: int = 32
    bio_channels: int = 3
    bio_window: int = 16
    dim: int = 256
    heads: int = 8
    layers: int = 4
    conv_kernels: tuple[int, ...] = (3, 5, 7)
    dropout: float = 0.1
    use_temporal: bool = True
    use_fusion: bool = True
    use_rl: bool = True
    use_recalibration: bool = True


@dataclass(frozen=True)
class DataConfig:
    horizon: int = 6
    n_train: int = 512
    n_val: int = 128
    n_test: int = 256
    temporal_window: int = 0
    dropped_channels: tuple[str, ...] = ()
    dataset: str = "nscisc"
    seed: int = 42


@dataclass(frozen=True)
class RewardConfig:
    w_pred: float = 0.3
    w_trans: float = 0.2
    w_cal: float = 0.2
    w_safe: float = 0.3
    under_triage_lambda: float = 2.0
    cal_window: int = 50


@dataclass(frozen=True)
class RqsConfig:
    lambda_cal: float = 0.5
    lambda_under: float = 1.0
    lambda_trans: float = 0.3


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 5e-4
    batch_size: int = 256
    epochs: int = 200
    patience: int = 20
    gamma: float = 0.99
    tau: float = 0.005
    cql_alpha: float = 0.1
    pretrain_epochs: int = 20
    mask_ratio: float = 0.15
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    warmup: int = 0
    amp: bool = False
    device: str = "cpu"
    seeds: tuple[int, ...] = (
        42,
        123,
        256,
        389,
        512,
        678,
        741,
        853,
        927,
        1024,
        1111,
        1337,
        1500,
        1776,
        2025,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "main"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rqs: RqsConfig = field(default_factory=RqsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


_SECTIONS: dict[str, type] = {
    "model": ModelConfig,
    "data": DataConfig,
    "reward": RewardConfig,
    "rqs": RqsConfig,
    "train": TrainConfig,
}


def _coerce(annotation: Any, value: Any) -> Any:
    origin = getattr(annotation, "__origin__", None)
    if origin is tuple:
        item_type = annotation.__args__[0]
        parts = [p for p in value.split(",") if p != ""] if isinstance(value, str) else list(value)
        return tuple(_coerce(item_type, p) for p in parts)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is str:
        return str(value)
    return value


def _update_section(instance: Any, updates: dict[str, Any]) -> Any:
    hints = get_type_hints(type(instance))
    coerced: dict[str, Any] = {}
    for key, raw in updates.items():
        if key not in hints:
            raise KeyError(f"unknown config key '{key}' for {type(instance).__name__}")
        coerced[key] = _coerce(hints[key], raw)
    return dataclasses.replace(instance, **coerced)


def _apply_mapping(base: ExperimentConfig, mapping: dict[str, Any]) -> ExperimentConfig:
    result = base
    if "name" in mapping:
        result = dataclasses.replace(result, name=str(mapping["name"]))
    for section in _SECTIONS:
        if section in mapping:
            current = getattr(result, section)
            updated = _update_section(current, dict(mapping[section]))
            result = dataclasses.replace(result, **{section: updated})
    return result


def parse_overrides(tokens: list[str]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"override '{token}' must be key=value")
        key, value = token.split("=", 1)
        parts = key.split(".")
        if len(parts) == 1:
            nested[parts[0]] = value
        elif len(parts) == 2:
            nested.setdefault(parts[0], {})[parts[1]] = value
        else:
            raise ValueError(f"override key '{key}' has too many levels")
    return nested


def load_experiment(
    leg: str | Path | None = None,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    config = ExperimentConfig()
    if leg is not None:
        path = _resolve_leg(leg)
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        config = _apply_mapping(config, data)
    if overrides:
        config = _apply_mapping(config, parse_overrides(overrides))
    return config


def _resolve_leg(leg: str | Path) -> Path:
    candidate = Path(leg)
    if candidate.exists():
        return candidate
    root = Path(__file__).resolve().parents[1] / "legs"
    named = root / f"{leg}.toml"
    if named.exists():
        return named
    raise FileNotFoundError(f"no leg config found for '{leg}'")
