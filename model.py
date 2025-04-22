import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention_weights = None  # To store weights for visualization

    def forward(self, x):
        B, T, D = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        self.attention_weights = attn.detach()

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)

class MiniTransformerBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.attn(x) + x
        return self.norm(x), self.attn.attention_weights
