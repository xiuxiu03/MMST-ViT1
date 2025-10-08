from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
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

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 使用可学习的滞后权重，但初始化为指数衰减（物理先验）
        self.lag_weights = nn.Parameter(torch.zeros(heads, max_time_lag + 1))
        with torch.no_grad():
            lags = torch.arange(max_time_lag + 1, dtype=torch.float)
            # 初始为指数衰减：滞后越久，权重越小
            initial_weights = torch.exp(-lags / (max_time_lag / 3))
            self.lag_weights.data = initial_weights.log().unsqueeze(0).expand(heads, -1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        h = self.heads
        b, t, _ = x.shape
        context = default(context, x)

        q = self.to_q(x)   # (b, t, inner_dim)
        k = self.to_k(context)  # (b, t, inner_dim)
        v = self.to_v(context)  # (b, t, inner_dim)

        # 重排为多头: (b, h, t, d)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), (q, k, v))

        # 计算相似度: (b, h, t, t)
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # 添加因果掩码（只能看过去）
        causal_mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()  # 上三角
        sim.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), max_neg_value(sim))

        # 构建时间滞后矩阵 (t, t): i-j 表示 query time i 可以看到 key time j 的滞后
        time_idx = torch.arange(t, device=x.device)
        time_lags = (time_idx.unsqueeze(1) - time_idx.unsqueeze(0))  # (t, t), 值为 i - j
        time_lags = time_lags.clamp(min=0, max=self.max_time_lag).long()  # 截断到 [0, max_lag]

        # lag_probs: (h, max_lag+1), 归一化
        lag_probs = F.softmax(self.lag_weights, dim=-1)  # (h, max_lag+1)

        # 正确查表：在 max_lag+1 维度上 gather
        lag_probs_expanded = lag_probs.unsqueeze(-1).unsqueeze(-1)  # (h, max_lag+1, 1, 1)
        indices = time_lags.unsqueeze(0).unsqueeze(-1).expand(h, t, t, 1)  # (h, t, t, 1)
        # 注意：gather 的 dim=1，因为我们是在 max_lag+1 这个维度查
        lag_adjustment = lag_probs_expanded.gather(1, indices)  # (h, t, t, 1)
        lag_adjustment = lag_adjustment.squeeze(-1)  # (h, t, t)
        lag_adjustment = lag_adjustment.unsqueeze(0)  # (1, h, t, t)

        # 加到注意力分数上（对数空间加等价于权重乘）
        sim = sim + lag_adjustment

        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.to_out(out)


class MultiModalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, context_dim=9, mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiModalAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)),
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
    def __init__(self, dim, depth, heads, dim_head, max_time_lag=5, dropout=0., max_seq_len=100):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        
        # 添加可学习时间位置编码
        self.time_pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        nn.init.normal_(self.time_pos_emb, std=0.02)

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

    def forward(self, x, context=None):
        # 加上时间位置编码
        b, t, _ = x.shape
        x = x + self.time_pos_emb[:, :t, :]
        if context is not None:
            context = context + self.time_pos_emb[:, :t, :]

        for attn, ff in self.layers:
            # 显式支持 context 输入
            x = attn(x, context=context) + x
            x = ff(x) + x
        return self.norm(x), None  # 返回 None 以兼容旧接口
