import models_pvt
from attention import TimeShiftedMultiModalAttention
from torch import nn
from einops import rearrange

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
        super(PVTSimCLR, self).__init__()
        
        self.backbone = models_pvt.__dict__[base_model](pretrained=pretrained)
        num_ftrs = self.backbone.head.in_features
        
        self.proj = nn.Linear(num_ftrs, out_dim)
        self.proj_context = nn.Linear(context_dim, out_dim)
        self.norm1 = nn.LayerNorm(context_dim)
        
        dim_head = out_dim // num_head
        self.mm_transformer = nn.ModuleList([])
        self.ff = nn.ModuleList([])
        
        for _ in range(mm_depth):
            # 时间偏移多模态注意力层
            self.mm_transformer.append(
                PreNorm(out_dim, TimeShiftedMultiModalAttention(
                    query_dim=out_dim,
                    context_dim=out_dim,
                    heads=num_head,
                    dim_head=dim_head,
                    max_time_lag=max_time_lag,
                    dropout=dropout
                ))
            )
            # 前馈网络层
            self.ff.append(
                PreNorm(out_dim, FeedForward(out_dim, dropout=dropout))
            )

    def forward(self, x, context=None, mask=None):
        # 视觉特征提取
        h = self.backbone.forward_features(x)  # [B, N, D]
        h = h.mean(dim=1) if h.dim() == 3 else h  # 确保[B, D]
        
        # 投影到目标维度
        x_visual = self.proj(h).unsqueeze(1)  # [B, 1, D]
        
        # 处理上下文信息
        if context is not None:
            context = self.proj_context(self.norm1(context))  # [B, T, D]
        else:
            # 如果没有提供上下文，使用视觉特征作为上下文
            context = x_visual.expand(-1, 1, -1)  # [B, 1, D]
        
        # 准备mask（如果需要）
        attn_mask = None
        if mask is not None:
            # 确保mask形状正确 [B, T]
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(x_visual.size(0), -1)
            attn_mask = mask
        
        # 多模态时间延迟注意力
        x_out = x_visual
        for attn_layer, ff_layer in zip(self.mm_transformer, self.ff):
            # 注意力层
            x_attn, _ = attn_layer(x_out, context=context, mask=attn_mask)
            x_out = x_attn + x_out  # 残差连接
            
            # 前馈层
            x_ff = ff_layer(x_out)
            x_out = x_ff + x_out  # 残差连接
        
        return x_out.squeeze(1)  # [B, D]
