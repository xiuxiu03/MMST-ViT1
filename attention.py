from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

def exists(val):
    return val is not None

def uniq(arr):
    return {el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# 辅助模块
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 时间延迟多模态注意力
class TimeShiftedMultiModalAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, max_time_lag=3, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.max_time_lag = max_time_lag
        
        # 可学习的滞后权重参数
        self.lag_weights = nn.Parameter(torch.randn(max_time_lag + 1))
        
        # 标准QKV投影
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 重排为多头形式
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> (b h) t d', h=h), (q, k, v))
        
        # 计算原始注意力分数
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        # 时间滞后掩码（仅允许访问当前及之前时刻）
        time_mask = torch.ones(sim.shape[-2:], device=x.device).triu(diagonal=1).bool()
        sim.masked_fill_(time_mask, -torch.finfo(sim.dtype).max)
        
        # 限制最大滞后步长
        if self.max_time_lag < sim.size(-1) - 1:
            lag_limit_mask = torch.ones_like(sim, device=x.device).tril(diagonal=-self.max_time_lag-1).bool()
            sim.masked_fill_(lag_limit_mask, -torch.finfo(sim.dtype).max)
        
        # 创建时间滞后索引 - 关键修正部分
        rows = torch.arange(sim.size(-2), device=x.device).view(-1, 1)
        cols = torch.arange(sim.size(-1), device=x.device).view(1, -1)
        time_lags = (rows - cols).clamp(min=0, max=self.max_time_lag)
        
        # 正确扩展lag_weights
        lag_weights = self.lag_weights.view(1, 1, -1)  # [1, 1, max_time_lag+1]
        # 修正expand调用方式
        lag_weights = lag_weights.expand(sim.size(0), sim.size(1), sim.size(2), -1)  # [B*H, T, T, max_time_lag+1]
        
        # 准备gather的索引
        time_lags_expanded = time_lags.unsqueeze(0).expand(sim.size(0), -1, -1).unsqueeze(-1)  # [B*H, T, T, 1]
        
        # 应用滞后权重
        lag_effect = torch.gather(
            lag_weights,
            dim=-1,
            index=time_lags_expanded
        ).squeeze(-1)  # [B*H, T, T]
        
        sim = sim + lag_effect
        
        # 应用输入mask（如有）
        if exists(mask):
            mask = rearrange(mask, 'b t -> (b h) 1 t', h=h)
            sim.masked_fill_(~mask, -torch.finfo(sim.dtype).max)
        
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) t d -> b t (h d)', h=h)
        return self.to_out(out), attn.detach()
        
# 空间注意力（保持不变）
class SpatialAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

# 时间注意力（保持不变）
class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, bias=None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(bias):
            bias = self.to_qkv(bias).chunk(3, dim=-1)
            qb, kb, _ = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), bias)
            bias = einsum('b h i d, b h j d -> b h i j', qb, kb) * self.scale
            dots += bias

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

# 修改后的多模态Transformer
class MultiModalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, context_dim=9, max_time_lag=3, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TimeShiftedMultiModalAttention(
                    dim, 
                    context_dim=context_dim, 
                    heads=heads, 
                    dim_head=dim_head,
                    max_time_lag=max_time_lag,
                    dropout=dropout
                )),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x, context=None, mask=None):
        attn_weights = []
        for attn, ff in self.layers:
            x_out, attn = attn(x, context=context, mask=mask)
            x = x_out + x
            x = ff(x) + x
            attn_weights.append(attn)
        return self.norm(x), attn_weights[-1]  # 返回最后一层注意力

# 空间Transformer（保持不变）
class SpatialTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# 时间Transformer（保持不变）
class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TemporalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x, bias=None):
        for attn, ff in self.layers:
            x = attn(x, bias=bias) + x
            x = ff(x) + x
        return self.norm(x)
