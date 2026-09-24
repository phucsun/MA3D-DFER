import torch
import torch.nn as nn
import torch.utils.checkpoint
from .MA3D import MA3D


class TemporalLSTM(nn.Module):
    """Bidirectional LSTM with linear projection to hidden_dim."""

    def __init__(self, input_dim=512, hidden_dim=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, features, seq_lengths=None):
        # features: (B, T, input_dim)
        lstm_out, _ = self.lstm(features)  # (B, T, hidden_dim*2)

        if seq_lengths is not None:
            B = features.shape[0]
            idx = (seq_lengths - 1).clamp(min=0).long()
            out = lstm_out[torch.arange(B, device=features.device), idx]
        else:
            out = lstm_out[:, -1]  # (B, hidden_dim*2)

        return self.fc(out)  # (B, hidden_dim)


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
    """
    Cải tiến:
      1. input_norm — ổn định feature từ frozen backbone.
      2. DropPath — stochastic depth regularization.
      3. CLS + masked mean pool fusion — biểu diễn phong phú hơn.
    """

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
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = embed_dim * 4

        # [NEW] Normalize feature từ backbone (quan trọng khi backbone frozen)
        self.input_norm = nn.LayerNorm(embed_dim)

        # DropPath rate tăng tuyến tính theo chiều sâu
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
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embed_dim)

        # [NEW] Fusion CLS + mean pool
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

        # [NEW] Input normalization
        features = self.input_norm(features)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)  # (B, T+1, D)
        features = features + self.pos_embedding[:, :T + 1]
        features = self.pos_drop(features)

        padding_mask = None
        if seq_lengths is not None:
            frame_mask = (
                torch.arange(T, device=features.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
            )
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=features.device)
            padding_mask = torch.cat((cls_mask, frame_mask), dim=1)

        for layer in self.layers:
            features = layer(features, src_key_padding_mask=padding_mask)

        out = self.norm(features)
        out_cls = out[:, 0]  # (B, D)

        # [NEW] Fuse với masked mean pool
        if self.use_cls_mean_fusion:
            frame_out = out[:, 1:]  # (B, T, D)
            if seq_lengths is not None:
                mask = torch.arange(T, device=frame_out.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
                mean_out = (frame_out * mask.unsqueeze(-1).float()).sum(1) \
                           / mask.sum(1, keepdim=True).float().clamp(min=1)
            else:
                mean_out = frame_out.mean(1)
            out_cls = self.fusion(torch.cat([out_cls, mean_out], dim=-1))

        return out_cls  # (B, 512)


class TemporalAttnPool(nn.Module):
    """BiLSTM + learned attention pooling thay cho last-state readout."""

    def __init__(self, input_dim=512, hidden_dim=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, features, seq_lengths=None):
        # features: (B, T, input_dim)
        lstm_out, _ = self.lstm(features)  # (B, T, hidden_dim*2)

        scores = self.attn(lstm_out).squeeze(-1)  # (B, T)

        if seq_lengths is not None:
            B, T = features.shape[:2]
            pad_mask = (
                torch.arange(T, device=features.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
            )
            scores = scores.masked_fill(pad_mask, float("-inf"))

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        out = (lstm_out * weights).sum(dim=1)  # (B, hidden_dim*2)

        return self.fc(out)  # (B, hidden_dim)


class TemporalMeanPool(nn.Module):
    """Masked mean pooling — baseline"""

    def forward(self, features, seq_lengths=None):
        if seq_lengths is not None:
            B, T = features.shape[:2]
            mask = torch.arange(T, device=features.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
            out = (features * mask.unsqueeze(-1).float()).sum(dim=1)
            out = out / mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        else:
            out = features.mean(dim=1)
        return out  # (B, embed_dim)


_TEMPORAL_MODULES = {
    "lstm": TemporalLSTM,
    "transformer": TemporalTransformer,
    "attn-pool": TemporalAttnPool,
    "mean": TemporalMeanPool,
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