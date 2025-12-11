# config.py
from dataclasses import dataclass, field, fields
from typing import Optional, Tuple, Union, List

import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


# ---------------- 子配置 ----------------

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    num_layers: int = 48
    d_model: int = 512
    num_heads: int = 25
    d_ff: int = 6400
    rope_theta: float = 2.0


@dataclass
class PathConfig:
    root: str = ""
    data: str = ""
    output: str = ""


@dataclass
class LRScheduleConfig:
    lr_max: float = 1e-3
    lr_min: float = 1e-5
    T_warmup: int = 10
    T_anneal: int = 10000


@dataclass
class ClippingConfig:
    max_l2_norm: float = 0.06
    eps: float = 1e-6


@dataclass
class OptimizerConfig:
    # 你在 yaml 里把 lr 注释掉了，这里给一个默认值，方便直接跑
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    eps: float = 1e-8


@dataclass
class TrainerConfig:
    epochs: int = 1
    batch_size: int = 64
    device: str = "cuda"
    z_loss: float = 0.0
    iters_per_epoch: int = 1000
    log_interval: int = 100
    save_interval: int = 1000


@dataclass
class DatasetConfig:
    train_data_path: str = "../data/TinyStories/TinyStoriesV2-GPT4-train.txt"
    # yaml 里写的是 1e4（float），这里改成 int 更合理
    train_data_shape: int = 10_000
    valid_data_path: str = "../data/TinyStories/TinyStoriesV2-GPT4-train.txt"
    # yaml 里写的是 1e4（float），这里改成 int 更合理
    valid_data_shape: int = 10_000


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "cs336-basics"
    entity: Optional[str] = None
    mode: str = "online"
    run_name: str = "${experiment_name}"  # 会用 experiment_name 覆盖


# ---------------- 根配置 ----------------

@dataclass
class Config:
    seed: int = 2025
    experiment_name: str = "baseline"

    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)
    clipping: ClippingConfig = field(default_factory=ClippingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class GenerationConfig:
    eos_token_id: Optional[List[int]] = None
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 512


# ---------------- 注册到 Hydra ----------------
def register_config():
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)


if __name__ == "__main__":
    # 关键一行：加上类型标注 → 补全立刻回来！

    cfg = OmegaConf.load(f"../../configs/generation_config.yaml")
    defaults = OmegaConf.structured(GenerationConfig)
    cfg_merged = OmegaConf.merge(defaults, cfg)
    generation_config: GenerationConfig = OmegaConf.to_object(cfg_merged)

    print(generation_config.top_p)
    print(generation_config.max_new_tokens)
    print(generation_config.eos_token_id)
