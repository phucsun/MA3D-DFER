import torch
import torch.nn as nn
import torch.utils.checkpoint
from .MA3D import MA3D


class TemporalLSTM(nn.Module):
    """Bidirectional LSTM với linear projection về hidden_dim."""

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


class TemporalTransformer(nn.Module):
    """Transformer encoder với masked mean pooling."""

    def __init__(self, embed_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, features, seq_lengths=None):
        # features: (B, T, embed_dim)
        padding_mask = None
        if seq_lengths is not None:
            B, T = features.shape[:2]
            # True = ignore this position
            padding_mask = (
                torch.arange(T, device=features.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
            )

        out = self.transformer(features, src_key_padding_mask=padding_mask)

        if seq_lengths is not None:
            valid_mask = ~padding_mask  # (B, T)
            out = (out * valid_mask.unsqueeze(-1).float()).sum(dim=1)
            out = out / valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        else:
            out = out.mean(dim=1)

        return out  # (B, embed_dim)


class TemporalMeanPool(nn.Module):
    """Masked mean pooling — baseline đơn giản nhất."""

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
    "mean": TemporalMeanPool,
}


class MA3D_Video(nn.Module):
    """
    Wrapper quanh MA3D backbone để nhận diện cảm xúc trên video.

    Pipeline:
        (B, T, 3, 224, 224)  →  per-frame MA3D features (B, T, 512)
                              →  temporal aggregation         (B, 512)
                              →  classification head          (B, num_classes)

    Args:
        img_size: kích thước ảnh đầu vào (mặc định 224)
        num_classes: số lớp cảm xúc (mặc định 7)
        type: biến thể backbone "small"|"base"|"large"
        use_3dmm: có dùng nhánh ThreeDMM hay không
        temporal_module: "lstm" | "transformer" | "mean"
        hidden_dim: chiều ẩn của temporal module (chỉ áp dụng với lstm)
        freeze_backbone: đóng băng toàn bộ MA3D trong giai đoạn 1
        head_dropout: dropout trước classification head (chống overfit)
        backbone_chunk_size: số frame tối đa mỗi lần forward qua backbone
    """

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
        elif temporal_module == "transformer":
            self.temporal = TemporalTransformer(embed_dim=512)
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
        """
        Args:
            video:      (B, T, 3, H, W)
            video_3d:   (B, T, 334) per-frame  hoặc  (B, 334) cố định  hoặc  None
            seq_lengths: (B,) số frame thực của mỗi video (trước khi padding)

        Returns:
            logits:   (B, num_classes)
            features: (B, hidden_dim)
        """
        B, T = video.shape[:2]

        # Batch tất cả T frames vào một forward pass duy nhất thay vì vòng lặp
        frames_flat = video.view(B * T, *video.shape[2:])  # (B*T, 3, H, W)

        x_3d_flat = None
        if self.use_3dmm and video_3d is not None:
            if video_3d.dim() == 3:        # (B, T, 334) — per-frame
                x_3d_flat = video_3d.reshape(B * T, video_3d.shape[-1])
            else:                          # (B, 334) — cố định cho cả video
                x_3d_flat = video_3d.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        # Forward backbone theo chunk thay vì đẩy toàn bộ B*T frame một lần.
        # Khi backbone được unfreeze, activations cho backward của cả B*T frame
        # vượt VRAM 12GB → Windows WDDM oversubscribe → driver reset (TDR) →
        # CUDA context chết, biểu hiện là cuDNN CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH.
        # Gradient checkpointing đổi ~30% compute lấy việc không phải giữ
        # activations của từng chunk (BN running stats bị update 2 lần/step —
        # sai lệch nhỏ, chấp nhận được).
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

        # (B, T, 512)
        frame_features = feat_flat.view(B, T, -1)

        temporal_feat = self.temporal(frame_features, seq_lengths)
        logits = self.head(self.dropout(temporal_feat))

        return logits, temporal_feat
