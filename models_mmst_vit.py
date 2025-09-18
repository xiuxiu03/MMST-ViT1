import torch
from torch import nn
from einops import rearrange, repeat

from attention import SpatialTransformer, TemporalTransformer
from models_pvt_simclr import PVTSimCLR


class MMST_ViT(nn.Module):
    def __init__(self, out_dim=2, num_grid=64, num_short_term_seq=6, num_long_term_seq=12, num_year=5,
                 pvt_backbone=None, context_dim=9, dim=192, batch_size=64, depth=4, heads=3, pool='cls', dim_head=64,
                 dropout=0., emb_dropout=0., scale_dim=4, max_time_lag=5):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.batch_size = batch_size
        self.pvt_backbone = pvt_backbone

        # 将多年长期气象数据映射到短期序列长度
        self.proj_context = nn.Linear(num_year * num_long_term_seq * context_dim, num_short_term_seq * dim)

        # pos_embedding 第三维 +1，因为拼接了 space_token
        # 原代码: num_grid -> num_grid+1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_short_term_seq, num_grid + 1, dim))

        # 空间 cls token
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))

        # 空间 Transformer
        self.space_transformer = SpatialTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)

        # 时间 cls token
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))

        # 初始化 TemporalTransformer 时添加 max_seq_len 参数
        self.temporal_transformer = TemporalTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            max_time_lag=max_time_lag,      # 滞后窗口大小
            dropout=dropout,
            max_seq_len=num_short_term_seq + 1  # 必须设置
        )

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        # 移除对 context 的单独 LayerNorm
        # 输出头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    def forward_features(self, x, ys):
        """
        使用 PVT 骨干网络提取每个网格的特征
        """
        # 重塑为 (batch*timesteps*grids, channels, height, width)
        x = rearrange(x, 'b t g c h w -> (b t g) c h w')
        ys = rearrange(ys, 'b t g n d -> (b t g) n d')

        B = x.shape[0]
        n = B // self.batch_size if B % self.batch_size == 0 else B // self.batch_size + 1
        x_hat = torch.empty(0).to(x.device)

        # 分批处理防止 OOM
        for i in range(n):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            x_tmp = x[start:end]
            ys_tmp = ys[start:end]
            x_hat_tmp = self.pvt_backbone(x_tmp, context=ys_tmp)
            x_hat = torch.cat([x_hat, x_hat_tmp], dim=0)

        return x_hat

    def forward(self, x, ys=None, yl=None):
        b, t, g, _, _, _ = x.shape

        # 提取空间特征
        x = self.forward_features(x, ys)
        x = rearrange(x, '(b t g) d -> b t g d', b=b, t=t, g=g)

        # 添加空间 cls token
        cls_space_tokens = repeat(self.space_token, '() g d -> b t g d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)  # 在 grid 维度拼接

        # 修改 4: pos_embedding 现在支持 g+1 个位置
        x += self.pos_embedding[:, :, :(g + 1)]  # 截取前 g+1 个位置编码
        x = self.dropout(x)

        # 展平 batch 和 time，进行空间 Transformer
        x = rearrange(x, 'b t g d -> (b t) g d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)  # 取出空间 cls token

        # 添加时间 cls token
        cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b t d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)  # 在时间维度拼接，形状: (b, t+1, dim)

        # 处理长期气象数据 yl
        yl = rearrange(yl, 'b y m d -> b (y m d)')  # 合并年、月、特征
        yl = self.proj_context(yl)  # 映射到 (b, t * dim)
        yl = rearrange(yl, 'b (t d) -> b t d', t=t)  # 重塑为 (b, t, dim)

        # 拼接时间 cls token 到气象数据
        yl = torch.cat((cls_temporal_tokens, yl), dim=1)  # 形状: (b, t+1, dim)

        # 将气象数据 yl 作为 context 传入 TemporalTransformer
        x = self.temporal_transformer(x, context=yl)  # 返回 (b, t+1, dim)

        # 池化：取 cls token 或平均
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # 输出预测
        return self.mlp_head(x)


if __name__ == "__main__":
    # 构造测试输入
    x = torch.randn((2, 6, 10, 3, 224, 224))      # 遥感序列: (B, T, G, C, H, W)
    ys = torch.randn((2, 6, 10, 28, 9))           # 短期气象: (B, T, G, N1, D)
    yl = torch.randn((2, 5, 12, 9))               # 长期气象: (B, Y, M, D)

    # 初始化 PVT 骨干网络
    pvt = PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9)

    # 初始化 MMST_ViT 模型
    model = MMST_ViT(
        out_dim=4,
        pvt_backbone=pvt,
        dim=512,
        num_short_term_seq=6,      # 必须传入，用于 max_seq_len
        max_time_lag=5             # 滞后窗口
    )

    # 前向传播
    z = model(x, ys=ys, yl=yl)
    print(z.shape)  # 期望输出: torch.Size([2, 4])
