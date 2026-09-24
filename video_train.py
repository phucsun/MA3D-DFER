import os
import time
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.MA3D_Video import MA3D_Video
from models.sam import SAM
from loss_function.loss import MarginAwareCELoss, LabelSmoothingCrossEntropy
from Read_dataset import VideoDataset, collate_video_fn
from video_engine import train_one_epoch_video, validate_video


class EarlyStopping:
    """
    Dừng training sớm nếu val_acc không cải thiện sau `patience` epochs.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = 0.0

    def step(self, val_acc: float) -> bool:
        """Trả về True nếu nên dừng."""
        if val_acc > self.best + self.min_delta:
            self.best = val_acc
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def get_args():
    parser = argparse.ArgumentParser("MA3D-Video Training")

    # Dataset
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Thư mục gốc dataset, phải có subfolder train/ và test/ (hoặc validation/)")
    parser.add_argument("--val_split", type=str, default=None,
                        help="Tên subfolder validation, vd 'test' hoặc 'validation' (mặc định: tự tìm)")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--stats_path", type=str, default=None,
                        help="File .npz chứa mean/std 3DMM (chỉ cần khi use_3dmm=True)")

    # 3DMM flag
    parser.add_argument("--use_3dmm", action="store_true",
                        help="Bật nhánh ThreeDMM. Mặc định TẮT.")
    parser.add_argument("--no_3dmm", dest="use_3dmm", action="store_false")
    parser.set_defaults(use_3dmm=False)

    # Model
    parser.add_argument("--model_type", default="large", choices=["small", "base", "large"])
    parser.add_argument("--temporal_module", default="lstm",
                        choices=["lstm", "transformer", "mean"])
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Đóng băng MA3D backbone trong giai đoạn 1 (mặc định True)")
    parser.add_argument("--unfreeze_backbone_epoch", type=int, default=None,
                        help="Epoch bắt đầu unfreeze backbone (None = không unfreeze)")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1,
                        help="Hệ số LR cho backbone so với head khi fine-tune (mặc định 0.1)")
    parser.add_argument("--head_dropout", type=float, default=0.3,
                        help="Dropout trước classification head")
    parser.add_argument("--backbone_checkpoint", type=str, default=None,
                        help="Path tới pretrained MA3D checkpoint (vd checkpoints/caers_MA3D.pth). "
                             "Nếu set, nạp vào backbone trước khi freeze.")

    # Video preprocessing
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Số frame tối đa mỗi video (None = không giới hạn)")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Lấy 1 frame mỗi N frames (temporal downsampling)")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Số DataLoader worker processes. Windows dùng 'spawn' mode "
                             "nên mỗi worker tốn ~1GB RAM; khuyến nghị 0-2.")

    # Mixed precision
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Dùng Automatic Mixed Precision (mặc định: bật)")
    parser.add_argument("--no_amp", dest="amp", action="store_false")

    # Early stopping
    parser.add_argument("--patience", type=int, default=10,
                        help="Số epochs không cải thiện trước khi dừng sớm (0 = tắt)")

    # Logging & checkpoint
    parser.add_argument("--log_file", type=str, default="video_log.txt")
    parser.add_argument("--resume_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_name", type=str, default="video_best.pth")
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def _find_val_split(data_dir: str, hint: str = None) -> str:
    """Tự tìm tên subfolder validation trong data_dir."""
    if hint:
        return hint
    for name in ("validation", "val", "test"):
        if os.path.isdir(os.path.join(data_dir, name)):
            return name
    raise FileNotFoundError(
        f"Không tìm thấy subfolder validation trong {data_dir}. "
        "Dùng --val_split để chỉ định rõ."
    )


def build_dataloaders(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    val_split = _find_val_split(args.data_dir, args.val_split)
    print(f"Validation split: '{val_split}'")

    train_dataset = VideoDataset(
        root=args.data_dir,
        split="train",
        transform=train_transform,
        use_3dmm=args.use_3dmm,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        stats_path=args.stats_path,
        clip_flip_p=0.5,
        random_temporal_crop=True,
    )
    val_dataset = VideoDataset(
        root=args.data_dir,
        split=val_split,
        transform=val_transform,
        use_3dmm=args.use_3dmm,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        stats_path=args.stats_path,
    )

    is_windows = os.name == "nt"
    persistent = (args.num_workers > 0) and (not is_windows)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
        collate_fn=collate_video_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        collate_fn=collate_video_fn,
    )
    return train_loader, val_loader


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.resume_dir, exist_ok=True)
    resume_path = os.path.join(args.resume_dir, args.resume_name)

    # Logging
    log_f = None
    if args.log_file:
        os.makedirs("log", exist_ok=True)
        log_path = os.path.join("log", args.log_file)
        log_f = open(log_path, "a")
        log_f.write(
            f"use_3dmm={args.use_3dmm}  temporal={args.temporal_module}  "
            f"amp={args.amp}  patience={args.patience}  batch={args.batch_size}\n"
        )
        log_f.write(f"checkpoint: {resume_path}\n")
        log_f.write(
            f"{'Epoch':^6} {'LR':^12} {'Train_Loss':^12} {'Train_Acc':^10} "
            f"{'Val_Loss':^12} {'Val_Acc':^10} {'Time(min)':^10} {'ES':^6}\n"
        )
        log_f.flush()

    train_loader, val_loader = build_dataloaders(args)

    model = MA3D_Video(
        num_classes=args.num_classes,
        type=args.model_type,
        use_3dmm=args.use_3dmm,
        temporal_module=args.temporal_module,
        hidden_dim=args.hidden_dim,
        freeze_backbone=args.freeze_backbone,
        head_dropout=args.head_dropout,
    ).to(device)

    if args.backbone_checkpoint and os.path.exists(args.backbone_checkpoint):
        ckpt = torch.load(args.backbone_checkpoint, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        print(f"Loaded backbone from {args.backbone_checkpoint}  "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    elif args.freeze_backbone:
        print(
            "WARNING: backbone đang bị freeze nhưng KHÔNG có --backbone_checkpoint — "
            "backbone sẽ giữ nguyên random init, features vô nghĩa và model chỉ có thể "
            "memorize. Hãy truyền --backbone_checkpoint hoặc --unfreeze_backbone_epoch."
        )

    CE_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)
    MA_criterion = MarginAwareCELoss().to(device)

    base_optimizer = optim.AdamW
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    backbone_params = list(model.backbone.parameters())
    optimizer = SAM(
        [
            {"params": head_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
        ],
        base_optimizer,
        lr=args.lr, weight_decay=args.weight_decay,
        rho=0.5, adaptive=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    early_stopper = EarlyStopping(patience=args.patience) if args.patience > 0 else None

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        if early_stopper and "es_counter" in ckpt:
            early_stopper.counter = ckpt["es_counter"]
            early_stopper.best = best_val_acc
        print(f"  epoch={start_epoch}  best_val_acc={best_val_acc:.4f}")

    epoch_bar = tqdm(
        range(start_epoch, args.epochs),
        desc="Epochs",
        unit="ep",
        initial=start_epoch,
        total=args.epochs,
        dynamic_ncols=True,
        leave=True,
    )

    for epoch in epoch_bar:
        if (
            args.unfreeze_backbone_epoch is not None
            and epoch == args.unfreeze_backbone_epoch
        ):
            model.unfreeze_backbone()
            tqdm.write(f"[Epoch {epoch + 1}] Backbone unfrozen.")

        t0 = time.time()

        train_loss, train_acc = train_one_epoch_video(
            model, train_loader,
            CE_criterion, lsce_criterion, MA_criterion,
            optimizer, device, epoch, args.epochs,
            use_3dmm=args.use_3dmm,
            use_amp=args.amp,
        )
        val_loss, val_acc = validate_video(
            model, val_loader, CE_criterion,
            device, epoch, args.epochs,
            use_3dmm=args.use_3dmm,
            use_amp=args.amp,
        )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = (time.time() - t0) / 60.0
        es_count = early_stopper.counter if early_stopper else 0

        epoch_bar.set_postfix({
            "lr":    f"{lr:.1e}",
            "t_acc": f"{train_acc * 100:.1f}%",
            "v_acc": f"{val_acc * 100:.1f}%",
            "best":  f"{best_val_acc * 100:.1f}%",
            "ES":    f"{es_count}/{args.patience}" if args.patience > 0 else "off",
        })

        summary = (
            f"Ep {epoch + 1:3d}/{args.epochs} | "
            f"lr={lr:.2e} | "
            f"train {train_loss:.4f}/{train_acc * 100:.2f}% | "
            f"val {val_loss:.4f}/{val_acc * 100:.2f}% | "
            f"{elapsed:.1f}min"
        )
        if val_acc >= best_val_acc:
            summary += "  ★ best"
        tqdm.write(summary)

        if log_f is not None:
            log_f.write(
                f"{epoch + 1:^6d} {lr:^12.8f} {train_loss:^12.4f} {train_acc * 100:^10.2f} "
                f"{val_loss:^12.4f} {val_acc * 100:^10.2f} {elapsed:^10.2f} "
                f"{es_count:^6d}\n"
            )
            log_f.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "es_counter": es_count,
                "args": vars(args),
            }, resume_path)
            if log_f is not None:
                log_f.write(f"BEST\tval_acc={best_val_acc * 100:.2f}\n")
                log_f.flush()

        if early_stopper and early_stopper.step(val_acc):
            tqdm.write(
                f"\nEarly stopping after epoch {epoch + 1} "
                f"(no improvement for {args.patience} epochs)."
            )
            if log_f is not None:
                log_f.write(f"EARLY_STOP\tepoch={epoch + 1}\n")
            break

    epoch_bar.close()

    print(f"\nBest validation accuracy: {best_val_acc * 100:.2f}%")
    if log_f is not None:
        log_f.write(f"\nBest validation accuracy: {best_val_acc * 100:.2f}%\n")
        log_f.close()


if __name__ == "__main__":
    main()
