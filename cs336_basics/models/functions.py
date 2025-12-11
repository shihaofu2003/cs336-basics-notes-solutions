from collections.abc import Iterator

import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce

import math


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x, _ = torch.max(x, dim=dim, keepdim=True)

    # Trick: Subtracting the maximum value in the i-th dimension
    x = x - max_x
    exp_x = torch.exp(x)
    res_softmax = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    return res_softmax


def siLU(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.exp(-x))


def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                                 mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k_sqrt = torch.sqrt(torch.tensor(keys.shape[-1]))
    # print("query: ", queries.device, "key: ", keys.device, "value: ", values.device)

    keys_T = rearrange(keys, "batch ... seq d_k -> batch ... d_k seq")
    scores = einsum(queries, keys_T, "batch ... seq_q d_k, batch ... d_k seq_k -> batch ... seq_q seq_k") / d_k_sqrt
    # print("score: ", scores.shape)

    if mask is not None:
        mask = mask.to(scores.device)
        scores = scores.masked_fill(~mask, -float('inf'))

    scores_norm = softmax(scores, dim=-1)
    # print("score: ", scores.device, "scores_norm: ", scores_norm.device, "value: ", values.device)

    return einsum(scores_norm, values, "batch ... seq_q seq_k, batch ... seq_k d_k -> batch ... seq_q d_k")


def cross_entropy(logits: torch.Tensor, target: torch.Tensor, z_loss: float = 0.0) -> torch.Tensor:
    # 由于在计算 softmax 时采用了 exp 操作，对于大型数据而言， exp 是指数爆炸的，导致数据在 exp 的倍数比原始倍数更大。
    # 可能导致某些数据在 softmax 后为 0，不适合作为 log 的输入

    logits_max, _ = logits.max(dim=-1, keepdim=True)
    logits = logits - logits_max

    logits = logits.reshape(-1, logits.size(-1))
    target = target.reshape(-1, 1)

    logits_target = logits.gather(dim=-1, index=target)
    logits_log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))

    cross_entropy_loss = reduce(logits_log_sum_exp - logits_target - z_loss * logits_log_sum_exp ** 2,
                                "batch ... -> ", reduction="mean")
    return cross_entropy_loss


def learning_rate_schedule(t: int, lr_max: float, lr_min: float, T_warmup: int, T_anneal: int) -> float:
    if t <= T_warmup:
        return lr_max * t / T_warmup
    elif T_warmup < t <= T_anneal:
        cos_part = math.cos(math.pi * (t - T_warmup) / (T_anneal - T_warmup)) + 1
        return lr_min + (lr_max - lr_min) * (cos_part / 2)
    else:
        return lr_min


def gradient_clipping(parameters: Iterator[torch.nn.parameter.Parameter],
                      max_l2_norm,
                      eps: float = 1e-6):

    # 把所有参数的梯度 看成一个大向量，计算 整体的 L2 范数
    norm_total_params = torch.tensor(0.0, dtype=torch.float32)
    for param in parameters:
        if param.grad is not None:
            norm_total_params += torch.norm(param.grad).item() ** 2
    norm_total_params = torch.sqrt(norm_total_params)

    if norm_total_params > max_l2_norm:
        clip_coef = max_l2_norm / (norm_total_params + eps)
        for param in parameters:
            if param.grad is None:
                continue
            param.grad = clip_coef * param.grad

    return parameters





if __name__ == "__main__":
    inputs = torch.arange(30).reshape(2, 3, 5).to(dtype=torch.float32)
    targets = torch.randint(5, (2, 3), dtype=torch.long)

    res = cross_entropy(inputs, targets)
    print(res)

    large_inputs = 1000.0 * inputs
    res = cross_entropy(large_inputs, targets)
    print(res)