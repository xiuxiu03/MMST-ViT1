import models_pvt
from attention import TimeShiftedMultiModalAttention  # 修改1：导入时间延迟注意力
from torch import nn
from einops import rearrange

# 新增辅助类
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim else dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class PVTSimCLR(nn.Module):
    def __init__(self, base_model, out_dim=512, context_dim=9, num_head=8, 
                 mm_depth=2, dropout=0., max_time_lag=3, pretrained=True):
        super().__init__()
        self.backbone = models_pvt.__dict__[base_model](pretrained=pretrained)
        num_ftrs = self.backbone.head.in_features
        
        # 投影层
        self.proj = nn.Linear(num_ftrs, out_dim)
        self.proj_context = nn.Linear(context_dim, out_dim)
        self.norm1 = nn.LayerNorm(context_dim)
        
        # 多模态Transformer（使用修改后的TimeShiftedMultiModalAttention）
        dim_head = out_dim // num_head
        self.mm_transformer = nn.ModuleList([
            PreNorm(out_dim, TimeShiftedMultiModalAttention(  # 依赖PreNorm
                query_dim=out_dim,
                context_dim=out_dim,
                heads=num_head,
                dim_head=dim_head,
                max_time_lag=max_time_lag,
                dropout=dropout
            )) for _ in range(mm_depth)
        ])
        self.ff = nn.ModuleList([
            PreNorm(out_dim, FeedForward(out_dim, dropout=dropout))  # 依赖FeedForward
            for _ in range(mm_depth)
        ])

    def forward(self, x, context=None, time_mask=None):
        # 视觉特征提取
        h = self.backbone.forward_features(x)
        h = h.mean(dim=1) if h.dim() == 3 else h
        
        # 投影
        x = self.proj(h).unsqueeze(1)  # [B, 1, D]
        context = self.proj_context(self.norm1(context))  # [B, T, D]
        
        # Transformer处理
        for attn, ff in zip(self.mm_transformer, self.ff):
            x_attn, _ = attn(x, context=context, mask=time_mask)
            x = x_attn + x  # 残差连接
            x = ff(x) + x    # 前馈网络
        
        return x.squeeze(1)  # [B, D]
