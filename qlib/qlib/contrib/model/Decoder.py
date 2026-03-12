import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [bs, L, d_model]
        return x + self.pe[:, : x.size(1), :].to(dtype=x.dtype)


class CrossAttentionDecoder(nn.Module):
    """
    多层“交叉注意力 + FFN”的 Transformer 示例：
    - 令时间序列特征 ts 作为 tgt（Query）
    - 令文本特征 txt 作为 memory（Key/Value）
    输入输出均为 [bs, context_len, d_model]
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        ts_pos_max_len: int = 4096,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.ts_pos = SinusoidalPositionalEncoding(d_model=d_model, max_len=ts_pos_max_len)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 直接用 [bs, L, d_model]
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, ts: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        # ts:  [bs, L_ts, d_model] 作为 Query 序列
        # txt: [bs, L_txt, d_model] 作为 Key/Value 序列
        ts = self.ts_pos(ts)
        return self.decoder(tgt=ts, memory=txt)  # [bs, L_ts, d_model]


# def main():
#     torch.manual_seed(0)

#     bs = 2
#     context_len = 16
#     d_model = 128

#     ts = torch.randn(bs, context_len, d_model)
#     txt = torch.randn(bs, context_len, d_model)

#     model = CrossAttentionTransformer(
#         d_model=d_model,
#         nhead=8,
#         num_layers=3,
#         dim_feedforward=4 * d_model,
#         dropout=0.1,
#     )

#     fused_ts = model(ts, txt)
#     print("ts shape:      ", tuple(ts.shape))
#     print("txt shape:     ", tuple(txt.shape))
#     print("fused_ts shape:", tuple(fused_ts.shape))

#     # 可选：如果你也想让文本反向融合时间序列（对称融合），再跑一次即可：
#     fused_txt = model(txt, ts)
#     print("fused_txt shape:", tuple(fused_txt.shape))


# if __name__ == "__main__":
#     main()
