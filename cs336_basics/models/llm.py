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

    def forward(self, x: torch.Tensor,  mask=None, token_positions=None):
        x_emb = self.embedding(x)

        for transformer_block in self.transformer_blocks:
            x_emb = transformer_block(x_emb, mask=mask, token_positions=token_positions)
        x_norm = self.rmsNorm(x_emb)

        y = self.linear_output(x_norm)
        return y

    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 pad_token_id: int = 256,
                 eos_token_id: int = 256,
                 do_sample: bool = False,
                 temperature: float = 1.,
                 repetition_penalty: float = 1.,
                 top_p: float = 0.9,
                 top_k: int = 1,
                 max_new_tokens: int = 512) -> torch.Tensor:

        batch, seq = input_ids.shape

        selected_eos = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
        selected_tokens: torch.Tensor = torch.tensor([], dtype=torch.int64, device=input_ids.device)
        for _ in range(min(max_new_tokens, self.context_length - seq)):
            logits = self.forward(input_ids, mask=attention_mask)[:, -1, :]  # (batch, vocab)

            # temperature
            logits_temp = logits / temperature

            # repetition_penalty
            if selected_tokens.numel() != 0:
                penalized = torch.gather(logits_temp, -1, selected_tokens)
                penalized[penalized > 0] = penalized[penalized > 0] / repetition_penalty
                penalized[penalized < 0] = penalized[penalized < 0] * repetition_penalty

                logits_temp.scatter_(-1, selected_tokens, penalized)

            # top-k
            if top_k > 1:
                top_k_value, _ = torch.topk(logits_temp, k=top_k, dim=-1)
                top_k_min_value = top_k_value[:, -1].unsqueeze(-1)  # (batch, 1)
                logits_temp = torch.where(logits_temp < top_k_min_value, -float('inf'), logits_temp)

            # top-p
            if top_p < 1:
                logits_sorted, indices_sorted = torch.sort(logits_temp, dim=-1, descending=True)
                logits_cumulative = softmax(logits_sorted, dim=-1).cumsum(dim=-1)

                mask = logits_cumulative >= top_p
                # 第一个 大于等于 top-p 的 token 也应该在候选集合内
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0  # 保留第一个 Token，防止没有 Token 选中

                mask = torch.scatter(mask, -1, indices_sorted, mask)
                logits_temp.masked_fill_(mask, -float('inf'))

            probs = softmax(logits_temp, -1)

            if do_sample:
                next_token_id = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            else:
                _, next_token_id = torch.max(probs, dim=-1, keepdim=True)

            selected_tokens = torch.concat((selected_tokens, next_token_id), dim=-1)
            input_ids = torch.concat((input_ids, next_token_id), dim=-1)

            # attn_mask from s*s -> (s+1) * (s+1)
            attn_mask_bottom = attention_mask[:, :, -1, :].unsqueeze(-2)
            attn_mask_left = torch.zeros((batch, 1, attention_mask.shape[-2] + 1, 1), dtype=torch.bool, device=input_ids.device)
            attn_mask_left[:, 0, -1, 0] = True

            attention_mask = torch.cat((attention_mask, attn_mask_bottom), dim=2)
            attention_mask = torch.cat((attention_mask, attn_mask_left), dim=3)

            # verify if generate all <eos> tokens
            selected_eos = selected_eos | (next_token_id == eos_token_id)
            if selected_eos.all():
                print("all sequences end with <eos>")
                break

        return input_ids










if __name__ == "__main__":
    model = TransformerLM(4, 512, 128,
                          1, 2, 512, 100)

    batch_size = 2
    seq_length = 12
    input_ids = torch.randint(4, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, 1, seq_length, seq_length), dtype=torch.bool)

    res = model.generate(input_ids, attention_mask, do_sample=True, eos_token_id=0, max_new_tokens=500)

    print(res.shape)



