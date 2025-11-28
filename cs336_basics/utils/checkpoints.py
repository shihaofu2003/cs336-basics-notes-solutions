import json

import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce

import os
from collections import defaultdict
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

from omegaconf import DictConfig, OmegaConf

from cs336_basics.utils.config import Config


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    state_dict = defaultdict()

    state_dict["model"] = model.state_dict()
    state_dict["optimizer"] = optimizer.state_dict()
    state_dict["iteration"] = iteration

    torch.save(state_dict, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:

    torch.serialization.add_safe_globals([defaultdict])
    state_dict = torch.load(src)

    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])

    if "optimizer" in state_dict:
        optimizer.load_state_dict(state_dict["optimizer"])

    if "iteration" in state_dict:
        return state_dict["iteration"]


def save_config_snapshot(cfg: DictConfig | Config, filename: str = "config_used.yaml"):
    """
    把本次实际使用的配置保存成一个 yaml 文件。
    由于 Hydra 会自动切到 run 目录，直接保存在当前目录即可。
    """
    print("save_config_snapshot: ", os.getcwd())
    OmegaConf.save(config=cfg, f=filename)

