import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce
import numpy.typing as npt
import numpy as np


def data_loading(dataset: npt.NDArray,
                 batch_size: int,
                 context_length: int,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.from_numpy(dataset).long()

    if data.shape[0] <= context_length:
        raise ValueError(f"Dataset must be longer than context_length. "
                         f"Now dataset shape: {data.shape} <= context_length: {context_length}")

    indices_start = torch.randint(low=0, high=data.shape[0]-context_length, size=(batch_size,))

    batch = torch.stack([data[index: index + context_length] for index in indices_start])
    target = torch.stack([data[index: index + context_length] for index in (indices_start + 1)])

    return batch.to(device=device), target.to(device=device)


class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, path, shape):
        self.data = np.memmap(path, dtype=np.int32, mode='r', shape=shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])  # 自动拷贝到 tensor
