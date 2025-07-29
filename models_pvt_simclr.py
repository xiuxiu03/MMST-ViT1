import models_pvt
from attention import TimeShiftedMultiModalAttention
from torch import nn


class PVTSimCLR(nn.Module):

    def __init__(self, base_model, out_dim=512, context_dim=9, num_head=8, mm_depth=2, dropout=0., pretrained=True, gated_ff=True):
        super(PVTSimCLR, self).__init__()

        self.backbone = models_pvt.__dict__[base_model](pretrained=pretrained)
        num_ftrs = self.backbone.head.in_features

        self.proj = nn.Linear(num_ftrs, out_dim)

        self.proj_context = nn.Linear(context_dim, out_dim)

        # 修改多模态Transformer为时间延迟版本
        dim_head = out_dim // num_head
        self.mm_transformer = nn.ModuleList([
            PreNorm(out_dim, TimeShiftedMultiModalAttention(
                query_dim=out_dim,
                context_dim=out_dim,
                heads=num_head,
                dim_head=dim_head,
                max_time_lag=max_time_lag,
                dropout=dropout
            )) for _ in range(mm_depth)
        ])
        self.ff = nn.ModuleList([
            PreNorm(out_dim, FeedForward(out_dim, dropout=dropout)) 
            for _ in range(mm_depth)
        ])

    def forward(self, x, context=None, time_mask=None):
    # 提取视觉特征
    h = self.backbone.forward_features(x)  # [B, N, D]
    h = h.squeeze()  # [B, D]
    
    # 投影到目标维度
    x = self.proj(h)  # [B, out_dim]
    context = self.proj_context(self.norm1(context))  # [B, T, out_dim]
    
    # 确保输入形状一致
    if x.dim() == 2:
        x = x.unsqueeze(1)  # [B, 1, out_dim]
    
    # 多模态时间延迟注意力
    for attn, ff in zip(self.mm_transformer, self.ff):
        x_attn, _ = attn(x, context=context, mask=time_mask)  # 传入时间掩码
        x = x_attn + x
        x = ff(x) + x
    
    # 返回分类token或全局平均
    return x.mean(dim=1) if x.size(1) > 1 else x.squeeze(1)
