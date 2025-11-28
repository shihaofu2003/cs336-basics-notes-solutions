from typing import Optional, Callable

import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), weight_decay=0.1, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)

        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()

        # self.param_groups 父类定义: List[Dict[str, Any]] = []
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                # self.state 父类定义: DefaultDict[torch.Tensor, Any] = defaultdict(dict)
                state = self.state[param]
                t = state.get("t", 1)

                grad = param.grad.data
                m = state.get("m", torch.zeros_like(grad, dtype=torch.float32))
                v = state.get("v", torch.zeros_like(grad, dtype=torch.float32))

                grad = param.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                param.data -= lr * weight_decay * param.data
                param.data -= lr * m_hat / torch.sqrt(v_hat + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


if __name__ == "__main__":
    model = torch.nn.Linear(5, 4)
    adamW = AdamW(model.parameters(), 0.1)
    print()



