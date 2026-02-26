from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


class RunConfig(BaseModel):
    name: str
    output_dir: str
    notes: Optional[str] = None


class DatasetConfig(BaseModel):
    root: str
    metadata_csv: str
    index_path: str
    image_size: int
    num_classes: int
    cache: bool = False
    cache_dir: str = ".cache"
    region_filter: Optional[str] = None
    time_filter: Optional[str] = None


class ModelConfig(BaseModel):
    name: str
    num_classes: int
    pretrained: bool = False


class TrainConfig(BaseModel):
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    mixed_precision: bool = False
    grad_clip: Optional[float] = None
    eval_every: int = 1
    save_every: int = 1
    fail_skip_batch: bool = True
    seed: int = 42


class LoggingConfig(BaseModel):
    log_interval: int = 10
    jsonl_path: str = "metrics.jsonl"
    sqlite_path: str = "runs.db"


class ArtifactsConfig(BaseModel):
    save_plots: bool = True
    confusion_matrix: bool = True


class ProfilingConfig(BaseModel):
    enabled: bool = True
    gpu_stats: bool = False


class DistributedConfig(BaseModel):
    enabled: bool = False
    backend: str = "nccl"
    shard_id: int = 0
    num_shards: int = 1


class AppConfig(BaseModel):
    run: RunConfig
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig
    logging: LoggingConfig
    artifacts: ArtifactsConfig
    profiling: ProfilingConfig
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)


@dataclass
class ConfigLoadResult:
    config: AppConfig
    raw: dict[str, Any]
    path: Path


def load_config(path: str | Path) -> ConfigLoadResult:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    try:
        cfg = AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise SystemExit(f"Invalid config: {exc}") from exc
    return ConfigLoadResult(config=cfg, raw=raw, path=path)


def snapshot_config(raw: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False))
