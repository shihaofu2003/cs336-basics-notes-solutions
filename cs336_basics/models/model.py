import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce

from cs336_basics.models.functions import softmax, scaled_dot_product_attention, siLU


class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        std_linear_weight = torch.sqrt(torch.tensor(2 / (self.in_features + self.out_features)))
        self.weight = nn.Parameter(data=torch.empty(size=(self.out_features, self.in_features),
                                                    dtype=self.dtype, device=self.device))
        torch.nn.init.trunc_normal_(tensor=self.weight, mean=0, std=std_linear_weight.item(),
                                    a=-3 * std_linear_weight.item(), b=3 * std_linear_weight.item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(Embedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        std_embedding_weight = torch.tensor(1)
        self.weight = nn.Parameter(data=torch.empty(size=(self.num_embeddings, self.embedding_dim),
                                                    dtype=self.dtype, device=self.device))
        torch.nn.init.trunc_normal_(tensor=self.weight, mean=0, std=std_embedding_weight.item(),
                                    a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(RMSNorm, self).__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(data=torch.ones(size=(self.d_model,),
                                                   dtype=self.dtype, device=self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size, sequence_length, d_model
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS = torch.sqrt(
            reduce(x * x, "batch sequence d_model -> batch sequence 1", "sum") / self.d_model
            + self.eps)
        result = x * self.weight / RMS

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(SwiGLU, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        if self.d_ff is None:
            self.d_ff = round(self.d_model * 8 / 3 / 64) * 64

        # print(d_ff, self.d_ff)
        self.w1 = Linear(self.d_model, self.d_ff, self.device, self.dtype)
        self.w3 = Linear(self.d_model, self.d_ff, self.device, self.dtype)
        self.w2 = Linear(self.d_ff, self.d_model, self.device, self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
        res_w1 = self.w1(x)
        silu = siLU(res_w1)
        return self.w2(silu * self.w3(x))


class RotaryPE_FullR(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super(RotaryPE_FullR, self).__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device("cpu")

        self._register_R()

    def _register_R(self):
        seq_col = torch.arange(self.max_seq_len, device=self.device).reshape(-1, 1).unsqueeze(2).unsqueeze(3)
        seq_col = seq_col.broadcast_to((self.max_seq_len, 1, 2, 2))
        inv_dk_row = self.theta ** (-torch.arange(0, self.d_k, 2, device=self.device) / self.d_k).reshape(1,
                                                                                                          -1).unsqueeze(
            2).unsqueeze(3)
        inv_dk_row = inv_dk_row.broadcast_to((1, self.d_k // 2, 2, 2))
        R_dk = seq_col * inv_dk_row
        # print("R_dk:", R_dk.shape)
        # print(R_dk[:, :, 0, 0].shape)
        R_dk[:, :, 0, 0] = torch.cos(R_dk[:, :, 0, 0])
        R_dk[:, :, 0, 1] = -torch.sin(R_dk[:, :, 0, 1])
        R_dk[:, :, 1, 0] = torch.sin(R_dk[:, :, 1, 0])
        R_dk[:, :, 1, 1] = torch.cos(R_dk[:, :, 1, 1])

        self.R = torch.zeros(self.max_seq_len, self.d_k, self.d_k, device=self.device)
        for i in range(self.d_k // 2):
            st = i * 2
            self.R[:, st:st + 2, st:st + 2] = R_dk[:, i, :, :]

        self.register_buffer("Rotation_Matrix", self.R, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        batch, seq_len, d_k = x.shape
        # print("x.shape: ", x.shape)
        return einsum(self.R[token_positions, :d_k, :d_k], x,
                      "seq_len d_k1 d_k, batch seq_len d_k -> batch seq_len d_k1")


class RotaryPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super(RotaryPE, self).__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        self._register_R()

    def _register_R(self):
        seq_col = torch.arange(self.max_seq_len).reshape(-1, 1)
        inv_dk_row = self.theta ** (-torch.arange(0, self.d_k, 2) / self.d_k).reshape(1, -1)
        R_dk = seq_col * inv_dk_row
        # print("R_dk:", R_dk.shape)

        self.R_sin = torch.sin(R_dk)
        self.R_cos = torch.cos(R_dk)

        self.register_buffer("Rotation_Matrix_sin", self.R_sin, persistent=False)
        self.register_buffer("Rotation_Matrix_cos", self.R_cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len, d_k = x.shape[-2:]
        # print("x.shape: ", x.shape)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        if token_positions is None:
            token_positions = torch.arange(0, seq_len)
        R_cos = self.R_cos[token_positions].to(x.device)
        R_sin = self.R_sin[token_positions].to(x.device)

        x_even_rotated = R_cos[:seq_len, :d_k] * x_even - R_sin[:seq_len, :d_k] * x_odd
        x_odd_rotated = R_sin[:seq_len, :d_k] * x_even + R_cos[:seq_len, :d_k] * x_odd

        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_even_rotated
        x_rotated[..., 1::2] = x_odd_rotated

        return x_rotated


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 rope: RotaryPE | None = None
                 ):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.linear_query = Linear(d_model, self.d_k * self.num_heads)
        self.linear_key = Linear(d_model, self.d_k * self.num_heads)
        self.linear_value = Linear(d_model, self.d_v * self.num_heads)

        self.linear_output = Linear(self.d_v * self.num_heads, self.d_model)

        self.rope = rope

    def forward(self,
                query_in: torch.Tensor,
                key_in: torch.Tensor,
                value_in: torch.Tensor,
                mask: torch.Tensor | None = None,
                token_positions: torch.Tensor | None = None
                ) -> torch.Tensor:

        queries = self.linear_query(query_in)
        keys = self.linear_key(key_in)
        values = self.linear_value(value_in)

        queries = rearrange(queries, "batch seq (head d_k) -> batch head seq d_k", head=self.num_heads)
        keys = rearrange(keys, "batch seq (head d_k) -> batch head seq d_k", head=self.num_heads)
        values = rearrange(values, "batch seq (head d_v) -> batch head seq d_v", head=self.num_heads)

        if self.rope is not None:
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        if mask is None:
            # Default: Causal masking
            mask = torch.tril(
                torch.ones((queries.shape[2], keys.shape[2]), dtype=torch.bool),
            )

        # print("mask:", mask.shape, queries.shape, keys.shape, values.shape)
        heads = scaled_dot_product_attention(queries, keys, values, mask)
        multi_head_attn = rearrange(heads, "batch head seq d_v -> batch seq (head d_v)", head=self.num_heads)

        return self.linear_output(multi_head_attn)


class Transformer(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 rope: RotaryPE | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.device = device
        self.dtype = dtype

        self.rmsNorm1 = RMSNorm(self.d_model, device=self.device)
        self.mha = MultiHeadAttention(self.d_model, self.num_heads, self.rope)

        self.rmsNorm2 = RMSNorm(self.d_model, device=self.device)
        self.swiGlu = SwiGLU(self.d_model, self.d_ff)

    def forward(self, x):
        x_norm = self.rmsNorm1(x)
        x = x + self.mha(x_norm, x_norm, x_norm)
        y = x + self.swiGlu(self.rmsNorm2(x))
        return y




if __name__ == '__main__':
    x = torch.randn(5, 16, 128)

    rope = RotaryPE(1, 128 // 4, 1024)
    transformer = Transformer(d_model=128, num_heads=4, d_ff=256, rope=rope)
    y = transformer(x)

    print(y.size())