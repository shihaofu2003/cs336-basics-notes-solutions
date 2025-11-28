import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce

from cs336_basics.models.model import Linear, Embedding, RMSNorm, SwiGLU, RotaryPE
from cs336_basics.models.model import MultiHeadAttention, Transformer
from cs336_basics.models.functions import softmax, scaled_dot_product_attention


class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float, ):
        super(TransformerLM, self).__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.embedding = Embedding(self.vocab_size, self.d_model)
        self.rope = RotaryPE(self.rope_theta, self.d_model // self.num_heads, self.context_length)
        self.transformer_blocks = nn.ModuleList(
            [Transformer(self.d_model, self.num_heads, self.d_ff, self.rope) for i in range(self.num_layers)])
        self.rmsNorm = RMSNorm(self.d_model)
        self.linear_output = Linear(self.d_model, self.vocab_size)

    def forward(self, x: torch.Tensor):
        x_emb = self.embedding(x)

        for transformer_block in self.transformer_blocks:
            x_emb = transformer_block(x_emb)
        x_norm = self.rmsNorm(x_emb)

        y = self.linear_output(x_norm)
        return y
        # return softmax(y, dim=-1)


def encode_lm(model: torch.nn.Module,
              tokens: torch.Tensor,
              temperature: float = 1.,
              top_p: float = 0.9,
              max_context_length: int = 512) -> torch.Tensor:

    if top_p > 1.0:
        raise ValueError("top_p can not be greater than 1.")




if __name__ == "__main__":
    pass
    # vocab_size: int = 1024
    # context_length: int = 2048
    # d_model: int = 256
    # num_layers: int = 2
    # num_heads: int = 4
    # d_ff: int = 512
    # rope_theta: float = 2
    #
    # x = torch.arange(16).reshape(-1, 16)
    # llm = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    #
    # y = llm.forward(x)
    # print(y.size())
