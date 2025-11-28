import random
import os
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import wandb

from cs336_basics.models.functions import cross_entropy, gradient_clipping, learning_rate_schedule
from cs336_basics.datasets.loading import data_loading
from cs336_basics.utils.config import Config
from cs336_basics.utils.checkpoints import load_checkpoint, save_checkpoint

def seed_set(seed=2025):
    """
    设置整个开发环境的随机种子
    :param seed: 随机种子数值
    """
    # 1. 设置 Python 原生模块的随机种子
    random.seed(seed)

    # 2. 设置 Python 哈希种子 (对字典等哈希结构的顺序有影响)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 3. 设置 Numpy 的随机种子
    np.random.seed(seed)

    # 4. 设置 PyTorch CPU 的随机种子
    torch.manual_seed(seed)

    # 5. 设置 PyTorch GPU 的随机种子 (如果是多GPU环境，使用 manual_seed_all)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 6. 设置 CuDNN 后端
    # deterministic=True 确保每次返回的卷积算法是确定的，但会牺牲速度
    torch.backends.cudnn.deterministic = True
    # benchmark=False 禁止自动寻找最优卷积算法，防止因不同输入尺寸导致算法选择不同从而带来随机性
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set as {seed}")


def wandb_set(cfg: DictConfig | Config) -> wandb.Run | None:
    run = None
    if cfg.wandb.enabled:
        # 把 DictConfig 转成普通 dict，方便 wandb 记录
        wandb_config = OmegaConf.to_container(cfg, resolve=True)

        wandb_kwargs = dict(
            project=cfg.wandb.project,
            config=wandb_config,
            name=cfg.wandb.run_name,
            mode=cfg.wandb.mode,
        )

        # 只有 entity 不为 null 时才传进去，避免权限问题
        if cfg.wandb.entity is not None:
            wandb_kwargs["entity"] = cfg.wandb.entity

        run = wandb.init(**wandb_kwargs, dir=cfg.paths.output)

    return run


def train_one_epoch(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    dataset,
                    run: wandb.Run | None,
                    cfg: Config,
                    device: torch.device):
    model.train()

    total_loss = 0.0

    for step in range(cfg.trainer.iters_per_epoch):
        # load_data
        x, y = data_loading(dataset, cfg.trainer.batch_size, cfg.model.context_length, device)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # print("x: ", x.device, "y: ", y.device)

        # forward_process
        optimizer.zero_grad()
        pred = model(x)
        loss = cross_entropy(pred, y)

        # backward_process + clipping + update
        loss.backward()

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.device, param.grad.device)
        #     else:
        #         print("⚠️ No grad for:", name)


        gradient_clipping(model.parameters(),
                          max_l2_norm=cfg.clipping.max_l2_norm, eps=cfg.clipping.eps)

        current_lr = learning_rate_schedule(step,
                                            lr_max=cfg.lr_schedule.lr_max,
                                            lr_min=cfg.lr_schedule.lr_min,
                                            T_warmup=cfg.lr_schedule.T_warmup,
                                            T_anneal=cfg.lr_schedule.T_anneal)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.step()

        total_loss += loss.item()

        if run is not None:
            wandb.log({"train/loss": total_loss/(step+1), "iter": step})

        if (step + 1) % cfg.trainer.log_interval == 0:
            print(f"Iter {step+1}, Avg_logg:{total_loss/(step+1): .2f}")

        if (step + 1) % cfg.trainer.save_interval == 0:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            save_checkpoint(model, optimizer, step,
                            out=f"checkpoints/{step}step_{total_loss/(step+1):.2f}.pt")

    avg_loss = total_loss / cfg.trainer.iters_per_epoch
    return avg_loss


if __name__ == '__main__':
    # 使用方法
    seed_set()
