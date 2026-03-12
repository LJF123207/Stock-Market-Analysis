import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionTSNews(nn.Module):
    def __init__(self, d_model, d_news, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        # Q 来自时序
        self.query_proj = nn.Linear(d_model, d_model)

        # K V 来自新闻
        self.key_proj = nn.Linear(d_news, d_model)
        self.value_proj = nn.Linear(d_news, d_model)

        head_dim = d_model // n_heads
        self.scale = head_dim ** 0.5

        # 残差 + LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, seq_features, news_features):
        """
        seq_features:  [B, T, d_model]
        news_features:[B, T, d_news]
        """

        B, T, _ = seq_features.size()
        head_dim = self.d_model // self.n_heads

        # Project
        Q = self.query_proj(seq_features)
        K = self.key_proj(news_features)
        V = self.value_proj(news_features)

        # reshape to multi-head
        Q = Q.view(B, T, self.n_heads, head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, head_dim).transpose(1, 2)

        # attention
        attn_scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, V)

        # merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 残差 + LN
        out = self.norm(seq_features + attn_output)

        return out, attn_weights
