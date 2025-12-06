
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from omegaconf import DictConfig, OmegaConf

from cs336_basics.utils.checkpoints import save_config_snapshot
from cs336_basics.utils.sets import seed_set, wandb_set, train_one_epoch
from cs336_basics.models.llm import TransformerLM
from cs336_basics.models.optim import AdamW
from cs336_basics.models.functions import cross_entropy
from cs336_basics.datasets.loading import MemmapDataset
from cs336_basics.utils.config import Config

from hydra.core.config_store import ConfigStore
# main.py 最上面
from pathlib import Path
import os

# 自动找到项目根目录（不管你在哪跑，都能找到 data/ 的位置）
ROOT_DIR = Path(__file__).resolve().parent  # 从 main.py 往上两层


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(
    version_base=None,
    config_path="configs",  # 配置文件所在目录
    config_name="config",  # 使用 configs/config.yaml
)
def main(cfg: Config):
    # 把路径塞进 cfg，后面随便用
    cfg.paths.root = str(ROOT_DIR)
    cfg.paths.data = str(ROOT_DIR / "data")
    cfg.paths.output = os.getcwd()  # Hydra 当前实验目录
    print("root: ", cfg.paths.root)
    print("data: ", cfg.paths.data)
    print("output: ", cfg.paths.output)

    # 当前工作目录已经是 hydra.run.dir 指定的目录
    current_dir = os.getcwd()
    print(f"Current working dir: {os.getcwd()}")
    print("Config:\n", OmegaConf.to_yaml(cfg))

    # 1. 设置随机种子
    seed_set(cfg.seed)

    # 2. 选择设备
    if cfg.trainer.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.trainer.device)

    # 3. 构建模型
    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta
    )

    model = model.to(device=device)

    # 4. 构建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        weight_decay=cfg.optimizer.weight_decay
    )

    train_dataset = np.load(cfg.dataset.train_data_path)
    valid_dataset = np.load(cfg.dataset.valid_data_path)

    # 5. 保存当前 config 副本（包括命令行 override 后的最终结果）
    save_config_snapshot(cfg, filename="config_used.yaml")

    # # 6. 初始化 wandb（如果开启）
    run = wandb_set(cfg)
    # run = None

    # 7. 训练循环
    for epoch in range(cfg.trainer.epochs):
        avg_loss = train_one_epoch(model, optimizer, train_dataset, valid_dataset, run, cfg, device)
        print(f"[Epoch {epoch}] loss = {avg_loss:.4f}")

        # log 到 wandb
        if run is not None:
            wandb.log({"train/loss": avg_loss, "epoch": epoch})

    # 8. 结束 wandb run
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
