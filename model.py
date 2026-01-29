import torch
from torch import nn
import torch.nn.functional as F


# ===============================================================
# 1. Directional Motion Map (magnitude)
# ===============================================================
class MotionMap(nn.Module):
    def __init__(self, num_frames_diff=2, eps=1e-6):
        super().__init__()
        self.num_frames_diff = num_frames_diff
        self.eps = eps

    def forward(self, x):

        """
        x: [B,T,H,W,C]
        Returns:
          motion_mag: [B,T,H,W,1]
        """

        B, T, H, W, C = x.shape
        pad = self.num_frames_diff
        x_padded = F.pad(x, (0,0,0,0,0,0,pad,0))  # pad temporal dim
        x_unfold = x_padded.unfold(1, self.num_frames_diff + 1, 1)  # [B,T,H,W,C,num_frames_diff+1]
        x_mean = x_unfold.mean(-1)
        flow = x - x_mean                     # signed temporal difference
        motion_mag = flow.norm(dim=-1, keepdim=True)
        motion_mag = motion_mag / (motion_mag.amax(dim=(2,3,4), keepdim=True) + self.eps)
        return motion_mag

# ===============================================================
# 2. Temporal Conditioned Conv (Multi-scale + Motion gating optional)
# ===============================================================
class TemporalConditionedConv(nn.Module):
    def __init__(self, dim, kernel=3, bottleneck_ratio=0.25, dropout=0.1,
                 multi_scale_tcc=True, use_motion_gating=True):
        super().__init__()
        self.multi_scale_tcc = multi_scale_tcc
        self.use_motion_gating = use_motion_gating

        bottleneck_dim = max(int(dim * bottleneck_ratio), 32)
        self.conv = nn.Conv2d(dim, bottleneck_dim, kernel, padding=kernel//2)
        self.temporal_fc = nn.Linear(dim * (3 if multi_scale_tcc else 1), bottleneck_dim)
        self.conv_out = nn.Conv2d(bottleneck_dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.res_scale = 0.5

        if use_motion_gating:
            self.gate_fc = nn.Linear(bottleneck_dim, 1)

    def forward(self, x, motion_map=None):

        """
        x: [B,T,H,W,C]
        motion_map: [B,T,H,W,1] (optional)
        """

        B, T, H, W, C = x.shape
        x_ = x.view(B*T, H, W, C).permute(0,3,1,2)
        y = self.conv(x_)

        # --- Temporal conditioning ---
        x_avg = x.mean(dim=(2,3))  # [B,T,C]
        if self.multi_scale_tcc:
            x_avg_2 = x.unfold(1, min(2, T), 1).mean(-1).mean(dim=(1, 2, 3))  # [B, C]
            x_avg_4 = x.unfold(1, min(4, T), 1).mean(-1).mean(dim=(1, 2, 3))  # [B, C]
            x_avg_multi = torch.cat([x_avg.mean(1), x_avg_2, x_avg_4], dim=-1)  # [B, C*3]
        else:
            x_avg_multi = x_avg.mean(1)

        weights = torch.sigmoid(self.temporal_fc(x_avg_multi)).view(B, -1, 1, 1)
        y = y * weights.repeat_interleave(T, dim=0)
        y = self.conv_out(y)
        y = self.dropout(y)
        y = y.permute(0,2,3,1).view(B,T,H,W,C)

        if self.use_motion_gating:
            motion_gate = motion_map.mean(dim=(2,3,4), keepdim=True) if motion_map is not None else 0
            g = torch.sigmoid(self.gate_fc(weights.mean(dim=(2,3)))[:, None, None, None, :] + motion_gate)
            return x + g * y * self.res_scale
        else:
            return x + y* self.res_scale


# ===============================================================
# Efficient Dynamic Attention with Linear Attention (Global + Local)
# ===============================================================
class EfficientDynamicAttention(nn.Module):
    def __init__(self, dim, heads=8, temporal_positional=True, dropout=0.1, window_size=None,
                  eps=1e-6):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.to_qkv = nn.Linear(dim, dim*3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.temporal_positional = temporal_positional
        self.window_size = window_size  # local attention window
        self.eps = eps

        if temporal_positional:
            self.temporal_emb = nn.Parameter(torch.randn(1, 1, 1, 1, dim))

    def feature_map(self, x):
        return F.elu(x) + 1  # positive kernel map for linear attention

    def linear_attention(self, q, k, v, motion_weight=None):
        """
        q, k, v: [B, heads, N, dim_head]
        motion_weight: [B, heads, N, 1] or None
        """
        q = self.feature_map(q)
        k = self.feature_map(k)

        if motion_weight is not None:
            q = q * motion_weight  # motion modulation

        k_sum = k.sum(-2, keepdim=True)  # [B, heads, 1, dim_head]
        kv = k.transpose(-2, -1) @ v     # [B, heads, dim_head, dim_head]
        out = q @ kv                     # [B, heads, N, dim_head]
        z = 1 / (q @ k_sum.transpose(-2,-1) + self.eps)
        return out * z

    def forward(self, x, motion_mag=None):
        B, T, H, W, C = x.shape

        if self.temporal_positional:
            x = x + self.temporal_emb

        # --- Temporal Attention ---
        x_flat_time = x.view(B, T, -1, C).mean(2)  # [B, T, C]
        qkv = self.to_qkv(x_flat_time).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.heads, self.dim_head).transpose(1,2) for t in qkv]

        motion_weight = None
        if motion_mag is not None:
            mm = motion_mag.view(B, T, -1).mean(-1, keepdim=True)
            motion_weight = 1 + torch.tanh(mm).unsqueeze(1)

        temporal_out = self.linear_attention(q, k, v, motion_weight)
        temporal_out = temporal_out.transpose(1,2).contiguous().view(B, T, C)
        temporal_out = temporal_out.unsqueeze(2).unsqueeze(2).expand(-1, -1, H, W, -1)
        x_spatial = x + temporal_out

        # --- Spatial Attention ---
        if self.window_size is None or self.window_size >= min(H, W):
            # Global linear attention
            x_flat = x_spatial.view(B * T, H * W, C)
            qkv = self.to_qkv(x_flat).chunk(3, dim=-1)
            q, k, v = [t.view(B*T, H*W, self.heads, self.dim_head).transpose(1,2) for t in qkv]

            motion_weight = None
            if motion_mag is not None :
                mm = motion_mag.view(B*T, H*W, 1)
                motion_weight = 1 + torch.tanh(mm).unsqueeze(1)

            spatial_out = self.linear_attention(q, k, v, motion_weight)
            spatial_out = spatial_out.transpose(1,2).contiguous().view(B, T, H, W, C)
        else:
            # Local window linear attention
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            x_pad = F.pad(x_spatial.permute(0,1,4,2,3), (0,pad_w,0,pad_h))  # [B,T,C,H_pad,W_pad]
            B, T, C, H_pad, W_pad = x_pad.shape
            H_win, W_win = H_pad // self.window_size, W_pad // self.window_size

            x_windows = x_pad.unfold(3, self.window_size, self.window_size).unfold(4, self.window_size, self.window_size)
            x_windows = x_windows.contiguous().view(B*T, C, H_win*W_win, self.window_size*self.window_size).permute(0,2,3,1)

            qkv = self.to_qkv(x_windows).chunk(3, dim=-1)
            q, k, v = [t.view(B*T, H_win*W_win, self.window_size*self.window_size, self.heads, self.dim_head).permute(0,3,1,2,4) for t in qkv]

            motion_weight = None
            if motion_mag is not None :
                motion_pad = F.pad(motion_mag.permute(0,1,4,2,3), (0,pad_w,0,pad_h))
                motion_windows = motion_pad.unfold(3, self.window_size, self.window_size).unfold(4, self.window_size, self.window_size)
                motion_windows = motion_windows.contiguous().view(B*T, H_win*W_win, self.window_size*self.window_size, 1)
                motion_weight = 1 + torch.tanh(motion_windows).unsqueeze(1)

            # Apply linear attention in local windows
            out_windows = self.linear_attention(q, k, v, motion_weight)  # [B*T, heads, H_win*W_win, window*window, dim_head]
            out_windows = out_windows.permute(0,2,3,1,4).contiguous().view(B*T, H_win*W_win, self.window_size*self.window_size, C)
            spatial_out = out_windows.view(B, T, H_pad, W_pad, C)[:, :, :H, :W, :]

        return spatial_out

# ===============================================================
# 5. Efficient ADAM Block
# ===============================================================
class EfficientADAMBlock(nn.Module):
    def __init__(self, dim, heads=8, bottleneck_ratio=0.25, dropout=0.3, window_size=None,
                 multi_scale_tcc=True, use_motion_gating=True):
        super().__init__()
        self.tcc = TemporalConditionedConv(
            dim, bottleneck_ratio=bottleneck_ratio, dropout=dropout,
            multi_scale_tcc=multi_scale_tcc, use_motion_gating=use_motion_gating
        )
        self.attn = EfficientDynamicAttention(
            dim, heads=heads, dropout=0.2, window_size=window_size
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, motion_mag=None):
        x = self.tcc(x, motion_mag)
        x = self.attn(x, motion_mag)
        x = x + self.ff(x)
        return x


# ===============================================================
# 6. Motion-Aware Temporal Pooling
# ===============================================================
class MotionAwareTemporalPooling(nn.Module):
    def __init__(self, dim, num_queries=8):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(1, num_queries, dim))

    def forward(self, x, motion_map=None):
      
        """
        x: [B, T, H, W, C]
        motion_map: [B, T, H, W, 1] or None
        """

        B,T,H,W,C = x.shape
        x_flat = x.view(B, T, H*W, C)
        if motion_map is not None:
            motion_flat = motion_map.view(B, T, H*W, 1)
            weights = motion_flat / (motion_flat.sum(dim=2, keepdim=True) + 1e-6)
            x_weighted = (x_flat * weights).sum(dim=2)
        else:
            x_weighted = x_flat.mean(dim=2)   # [B, T, C]
        query = self.query.expand(B, -1, -1)  # [B, Q, C]
        attn = (query @ x_weighted.transpose(1,2)).softmax(-1)  # [B, Q, T]
        pooled = attn @ x_weighted   # [B, Q, C]
        return pooled.view(B, -1)


# ===============================================================
# 7. Full Model with Configurable Features
# ===============================================================
class Model(nn.Module):
    def __init__(self, input_dim=192, dim=32, num_blocks=1, heads=8, dropout=0.4,
                 num_queries=8, window_size=None,
                 multi_scale_tcc=True,
                 use_motion_gating=True,
                 motion=True):
        super().__init__()
        self.input_dim = input_dim * 2
        self.norm0 = nn.LayerNorm(self.input_dim)
        self.input_proj = nn.Linear(self.input_dim, dim)
        self.motion_map_module = MotionMap(num_frames_diff=2)
        self.blocks = nn.ModuleList([
            EfficientADAMBlock(dim, heads=heads, dropout=dropout, window_size=window_size,
                               multi_scale_tcc=multi_scale_tcc,
                               use_motion_gating=use_motion_gating)
            for _ in range(num_blocks)
        ])
        self.temporal_pool = MotionAwareTemporalPooling(dim, num_queries=num_queries)
        #self.temporal_pool = nn.MaxPool3d(kernel_size=(5,5,8), stride=(5,5,8))
        self.norm = nn.LayerNorm(dim*num_queries)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim*num_queries, 1)
        self.motion=motion

    def forward(self, x):
        B,C,T,H,W = x.shape

        # motion difference features
        motion_feat = (x[:,:,1:,:,:] - x[:,:,:-1,:,:]).abs()
        motion_feat = F.pad(motion_feat, (0,0,0,0,1,0))
        x = torch.cat([x, motion_feat.abs()], dim=1)

        x = x.permute(0,2,3,4,1)
        x = self.norm0(x)
        x = self.input_proj(x)

        if self.motion==True:
          motion_mag= self.motion_map_module(x)
        else:
          motion_mag=None

        for block in self.blocks:
            x = block(x, motion_mag)

        pooled = self.temporal_pool(x, motion_mag)
        #pooled = self.temporal_pool(x)
        #pooled = pooled.view(B, -1)
        pooled = self.norm(self.dropout(pooled))
        logits = self.fc(pooled)
        probs = torch.sigmoid(logits)
        return probs, pooled