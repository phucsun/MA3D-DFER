import torch
import torch.nn as nn
import torch.utils.checkpoint
from .MA3D import MA3D

class FrequencyTokenizer(nn.Module):
    def __init__(self,
                 embed_dim=512,
                 num_freq_tokens=4):
        super().__init__()

        self.num_freq_tokens = num_freq_tokens

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        """
        x : (B,T,C)
        """

        # FFT theo chiều thời gian
        freq = torch.fft.rfft(x, dim=1)

        # magnitude
        freq = freq.abs()

        # chỉ giữ low-frequency
        freq = freq[:, :self.num_freq_tokens]

        freq = self.proj(freq)

        return freq

class DropPath(nn.Module):
    """Stochastic Depth — tắt ngẫu nhiên residual block trong lúc training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob).div_(keep_prob)
        return x * noise


class _TFLayerWithDropPath(nn.Module):
    """Wrapper: TransformerEncoderLayer + DropPath trên combined residual."""
    def __init__(self, base_layer: nn.TransformerEncoderLayer, drop_path_rate: float):
        super().__init__()
        self.base = base_layer
        self.dp = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        delta = self.base(x, src_key_padding_mask=src_key_padding_mask) - x
        return x + self.dp(delta)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 256,
        dim_feedforward: int = None,       
        drop_path_rate: float = 0.1,       
        use_cls_mean_fusion: bool = True,   
        num_freq_tokens=4,
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = embed_dim * 4

        self.input_norm = nn.LayerNorm(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            base = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                activation='gelu',
                norm_first=True,
                batch_first=True,
            )
            self.layers.append(_TFLayerWithDropPath(base, dpr[i]))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_freq_tokens = num_freq_tokens
        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_len + 1 + self.num_freq_tokens,
                embed_dim
            )
        )
        self.pos_drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embed_dim)

        self.freq_branch = FrequencyTokenizer(
            embed_dim=embed_dim,
            num_freq_tokens=self.num_freq_tokens
        )

        self.use_cls_mean_fusion = use_cls_mean_fusion
        if use_cls_mean_fusion:
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, features: torch.Tensor, seq_lengths=None) -> torch.Tensor:
        B, T, _ = features.shape

        features = self.input_norm(features)
        freq_tokens = self.freq_branch(features)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Chuỗi kết hợp: [CLS (1), Video_Frames (T), Freq_Tokens (num_freq)]
        features = torch.cat([cls_tokens, features, freq_tokens], dim=1)
        
        features = features + self.pos_embedding[:, :T + 1 + self.num_freq_tokens]
        features = self.pos_drop(features)

        padding_mask = None
        if seq_lengths is not None:
            frame_mask = (
                torch.arange(T, device=features.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
            )
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=features.device)
            freq_mask = torch.zeros((B, self.num_freq_tokens), dtype=torch.bool, device=features.device)

            padding_mask = torch.cat([cls_mask, frame_mask, freq_mask], dim=1)

        for layer in self.layers:
            features = layer(features, src_key_padding_mask=padding_mask)

        out = self.norm(features)
        out_cls = out[:, 0]  # Lấy token CLS tại index 0

        # [SỬA LỖI LOGIC TẠI ĐÂY]
        if self.use_cls_mean_fusion:
            # Chỉ cắt lấy các token tương ứng với video frames (từ index 1 đến T+1)
            frame_out = out[:, 1 : T + 1]  # Kích thước chuẩn: (B, T, D)
            
            if seq_lengths is not None:
                # Tạo mask hợp lệ cho các frame thực tế
                mask = torch.arange(T, device=frame_out.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
                # Masked mean pooling
                mean_out = (frame_out * mask.unsqueeze(-1).float()).sum(1) \
                           / mask.sum(1, keepdim=True).float().clamp(min=1)
            else:
                mean_out = frame_out.mean(1)
                
            # Đưa qua tầng fusion tuyến tính ban đầu mong muốn (B, D*2 -> B, D)
            out_cls = self.fusion(torch.cat([out_cls, mean_out], dim=-1))

        return out_cls
    
_TEMPORAL_MODULES = {
    # "lstm": TemporalLSTM,
    "transformer": TemporalTransformer,
    # "attn-pool": TemporalAttnPool,
    # "mean": TemporalMeanPool,
}


class MA3D_Video(nn.Module):
    """Wrapper MA3D backbone for video."""

    def __init__(
        self,
        img_size: int = 224,
        num_classes: int = 7,
        type: str = "large",
        use_3dmm: bool = False,
        temporal_module: str = "lstm",
        hidden_dim: int = 512,
        freeze_backbone: bool = True,
        head_dropout: float = 0.3,
        backbone_chunk_size: int = 64,
    ):
        super().__init__()
        self.use_3dmm = use_3dmm
        self.backbone_chunk_size = backbone_chunk_size

        self.backbone = MA3D(
            img_size=img_size,
            num_classes=num_classes,
            type=type,
            use_3dmm=use_3dmm,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if temporal_module == "lstm":
            self.temporal = TemporalLSTM(input_dim=512, hidden_dim=hidden_dim)
            out_dim = hidden_dim
        elif temporal_module == "attn-pool":
            self.temporal = TemporalAttnPool(input_dim=512, hidden_dim=hidden_dim)
            out_dim = hidden_dim
        elif temporal_module == "transformer":
            self.temporal = TemporalTransformer(
                embed_dim=512,
                num_layers=4,
                dim_feedforward=512 * 4,   
                drop_path_rate=0.1,       
                use_cls_mean_fusion=True,  
            )
            out_dim = 512
        elif temporal_module == "mean":
            self.temporal = TemporalMeanPool()
            out_dim = 512
        else:
            raise ValueError(f"Unknown temporal_module: {temporal_module!r}. Choose from {list(_TEMPORAL_MODULES)}")

        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(out_dim, num_classes)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, video, video_3d=None, seq_lengths=None):
        B, T = video.shape[:2]

        frames_flat = video.view(B * T, *video.shape[2:])  # (B*T, 3, H, W)

        x_3d_flat = None
        if self.use_3dmm and video_3d is not None:
            if video_3d.dim() == 3:
                x_3d_flat = video_3d.reshape(B * T, video_3d.shape[-1])
            else:
                x_3d_flat = video_3d.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        use_ckpt = self.training and backbone_trainable and torch.is_grad_enabled()

        chunk = self.backbone_chunk_size
        feat_chunks = []
        for start in range(0, B * T, chunk):
            frames_chunk = frames_flat[start:start + chunk]
            chunk_3d = x_3d_flat[start:start + chunk] if x_3d_flat is not None else None
            if use_ckpt:
                _, feat, _ = torch.utils.checkpoint.checkpoint(
                    self.backbone, frames_chunk, chunk_3d, use_reentrant=False
                )
            else:
                _, feat, _ = self.backbone(frames_chunk, chunk_3d)
            feat_chunks.append(feat)
        feat_flat = torch.cat(feat_chunks, dim=0)  # (B*T, 512)

        frame_features = feat_flat.view(B, T, -1)  # (B, T, 512)

        temporal_feat = self.temporal(frame_features, seq_lengths)
        logits = self.head(self.dropout(temporal_feat))

        return logits, temporal_feat