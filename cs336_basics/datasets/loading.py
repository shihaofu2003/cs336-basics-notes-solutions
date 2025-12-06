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


def valid_data_loading(dataset: npt.NDArray,
                       batch_size: int,
                       context_length: int,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    这是一个生成器，用于在验证阶段按顺序遍历整个数据集
    """
    data = torch.from_numpy(dataset).long()
    n_samples = data.shape[0]

    step = context_length
    indices = torch.arange(0, n_samples - context_length, step)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i: i + batch_size]

        current_batch_size = len(batch_indices)
        if current_batch_size == 0:
            break

        x_list = []
        y_list = []

        for start_idx in batch_indices:
            end_idx = start_idx + context_length
            x_list.append(data[start_idx: end_idx])
            y_list.append(data[start_idx + 1: end_idx + 1])

        batch = torch.stack(x_list)
        target = torch.stack(y_list)

        yield batch.to(device=device), target.to(device=device)


class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, path, shape):
        self.data = np.memmap(path, dtype=np.int32, mode='r', shape=shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])  # 自动拷贝到 tensor
