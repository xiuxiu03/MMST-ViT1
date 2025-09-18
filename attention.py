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


# feedforward
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


class MultiModalAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

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

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


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

class TimeShiftedCrossModalAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, max_time_lag=5, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.max_time_lag = max_time_lag

        # 遥感图像到查询向量
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # 气象数据到键值向量
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 可学习的滞后权重参数，每个头独立学习
        self.lag_weights = nn.Parameter(torch.randn(heads, max_time_lag + 1))
        
        # 输出层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        h = self.heads

        # 查询来自遥感图像
        q = self.to_q(x)
        
        # 键值来自气象数据
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # 重排为多头形式
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> (b h) t d', h=h), (q, k, v))

        # 计算原始注意力分数
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # 创建因果掩码 - 遥感图像只能访问当前及之前时刻的气象数据
        t = sim.size(1)
        causal_mask = torch.ones(t, t, device=x.device).triu_(1).bool()
        max_neg_val = -torch.finfo(sim.dtype).max
        sim.masked_fill_(causal_mask.unsqueeze(0), max_neg_val)

        # 应用时间滞后权重
        time_lags = torch.arange(t, device=x.device).view(1, -1, 1) - torch.arange(t, device=x.device).view(1, 1, -1)
        time_lags = time_lags.clamp(min=0, max=self.max_time_lag)
        
        # 扩展滞后权重到batch维度
        lag_effect = self.lag_weights.unsqueeze(0).unsqueeze(3)  # [1, heads, max_lag+1, 1]
        lag_effect = lag_effect.expand(sim.size(0), -1, -1, t)   # [batch*heads, heads, max_lag+1, t]
        
        # 为每个头选择对应的滞后权重
        head_indices = torch.arange(h, device=x.device).repeat(sim.size(0) // h)
        selected_lag_weights = lag_effect[torch.arange(sim.size(0)), head_indices]  # [batch*heads, max_lag+1, t]
        
        # 应用滞后效应
        lag_adjustment = selected_lag_weights.gather(1, time_lags.expand(sim.size(0), -1, -1))
        sim = sim + lag_adjustment

        # 注意力计算
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) t d -> b t (h d)', h=h)
        
        # 返回输出张量
        return self.to_out(out)

class MultiModalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, context_dim=9, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiModalAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head,
                                                 dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim_out=dim, mult=mult, dropout=dropout))
            ]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return self.norm(x)


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


class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, max_time_lag=5, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TimeShiftedCrossModalAttention(
                    query_dim=dim, 
                    context_dim=dim, 
                    heads=heads, 
                    dim_head=dim_head, 
                    max_time_lag=max_time_lag, 
                    dropout=dropout
                )),
                PreNorm(dim, FeedForward(dim, dropout=dropout))
            ]))

    def forward(self, x, bias=None, context=None):
        for attn, ff in self.layers:
            if context is not None:
                # 使用跨模态注意力，x作为query，context作为key/value
                x = attn(x, context=context) + x
            else:
                # 使用自注意力
                x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
