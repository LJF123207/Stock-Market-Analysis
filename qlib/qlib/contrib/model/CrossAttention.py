import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionTSNews(nn.Module):
    def __init__(self, d_model, d_news, n_heads=8, dropout=0.2, ffn_ratio=4, activation="gelu"):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # === Cross Attention ===
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_news, d_model)
        self.value_proj = nn.Linear(d_news, d_model)
        self.out_proj = nn.Linear(d_model, d_model)  

        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # === FFN ===
        hidden_dim = d_model * ffn_ratio
        if activation == "gelu":
            act_layer = nn.GELU()
        elif activation == "relu":
            act_layer = nn.ReLU()
        else:
            raise ValueError("activation must be gelu or relu")

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)  # 可选：输出前再加一次
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, seq_features, news_features):
        """
        seq_features: [B, T, d_model]
        news_features: [B, T, d_news]
        """
        B, T, _ = seq_features.shape

        # =============== 1. Cross-Attention (Pre-Norm) ===============
        x = self.norm1(seq_features)  # Pre-Norm

        Q = self.query_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(news_features).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(news_features).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [B, H, T, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        attn_output = self.out_proj(attn_output)

        # Residual
        x = seq_features + attn_output

        # =============== 2. FFN (Pre-Norm) ===============
        x_ffn = self.norm2(x)  # Pre-Norm
        ffn_out = self.ffn(x_ffn)
        out = x + ffn_out

        return out, attn_weights  # attn_weights: [B, H, T, T]