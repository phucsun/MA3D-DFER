import os
import time
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from models.MA3D_Video import MA3D_Video
from models.sam import SAM
from loss_function.loss import MarginAwareCELoss, LabelSmoothingCrossEntropy
from Read_dataset import VideoDataset, collate_video_fn
from video_engine import train_one_epoch_video, validate_video


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = 0.0

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best + self.min_delta:
            self.best = val_acc
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def get_args():
    parser = argparse.ArgumentParser("MA3D-Video Training")

    # Dataset
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--val_split", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--stats_path", type=str, default=None)

    # 3DMM flag
    parser.add_argument("--use_3dmm", action="store_true")
    parser.add_argument("--no_3dmm", dest="use_3dmm", action="store_false")
    parser.set_defaults(use_3dmm=False)

    # Model
    parser.add_argument("--model_type", default="large", choices=["small", "base", "large"])
    parser.add_argument("--temporal_module", default="lstm",
                        choices=["lstm", "transformer", "attn-pool", "mean"])
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze_backbone_epoch", type=int, default=None)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1)
    parser.add_argument("--head_dropout", type=float, default=0.3)
    parser.add_argument("--backbone_checkpoint", type=str, default=None)

    # Video preprocessing
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames per video (None = no limit)")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Temporal downsampling")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Số DataLoader worker processes. Windows dùng 'spawn' mode "
                             "nên mỗi worker tốn ~1GB RAM; khuyến nghị 0-2.")

    # Mixed precision
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")

    # Class imbalance (Strategy A) — cân bằng lớp thiểu số (Fear/Disgust)
    parser.add_argument("--use_class_weights", action="store_true", default=True,
                        help="Bật class-weighted loss (mặc định True)")
    parser.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--cb_loss_pow", type=float, default=0.5,
                        help="Số mũ cho inverse-frequency của loss weight (0=tắt, 0.5=nhẹ, 1=mạnh)")
    parser.add_argument("--use_weighted_sampler", action="store_true", default=True,
                        help="Bật WeightedRandomSampler để cân bằng batch (mặc định True)")
    parser.add_argument("--no_weighted_sampler", dest="use_weighted_sampler", action="store_false")
    parser.add_argument("--cb_sampler_pow", type=float, default=1.0,
                        help="Số mũ cho inverse-frequency của sampler weight (0=tắt, 1=cân bằng đầy đủ)")

    # Model selection (Strategy B)
    parser.add_argument("--select_metric", default="uar", choices=["uar", "war", "mean"],
                        help="Metric chọn best checkpoint & early stopping "
                             "(uar=mặc định, war=accuracy, mean=trung bình hai)")

    # Early stopping
    parser.add_argument("--patience", type=int, default=10)

    # Logging & checkpoint
    parser.add_argument("--log_file", type=str, default="video_log.txt")
    parser.add_argument("--resume_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_name", type=str, default="video_best.pth")
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def _find_val_split(data_dir: str, hint: str = None) -> str:
    if hint:
        return hint
    for name in ("validation", "val", "test"):
        if os.path.isdir(os.path.join(data_dir, name)):
            return name
    raise FileNotFoundError(
        f"Cannot find subfolder validation in {data_dir}. "
        "Use --val_split for specify folder."
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

    train_labels = torch.tensor([s["label"] for s in train_dataset.samples], dtype=torch.long)
    class_counts = torch.bincount(train_labels, minlength=args.num_classes).float()
    print("Train class counts: " + ", ".join(
        f"{c}={int(n)}" for c, n in zip(train_dataset.class_names, class_counts)
    ))

    sampler = None
    if args.use_weighted_sampler and args.cb_sampler_pow > 0:
        inv = (1.0 / class_counts.clamp(min=1)) ** args.cb_sampler_pow
        sample_weights = inv[train_labels]                       # (N,)
        sampler = WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=len(train_labels),
            replacement=True,
        )
        print(f"WeightedRandomSampler ON (pow={args.cb_sampler_pow})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),         
        sampler=sampler,
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
    return train_loader, val_loader, class_counts


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
            f"amp={args.amp}  patience={args.patience}  batch={args.batch_size}  "
            f"select={args.select_metric}  cls_w={args.use_class_weights}(pow={args.cb_loss_pow})  "
            f"sampler={args.use_weighted_sampler}(pow={args.cb_sampler_pow})\n"
        )
        log_f.write(f"checkpoint: {resume_path}\n")
        log_f.write(
            f"{'Epoch':^6} {'LR':^12} {'Train_Loss':^12} {'Train_Acc':^10} "
            f"{'Val_Loss':^12} {'Val_WAR':^10} {'Val_UAR':^10} {'Time(min)':^10} {'ES':^6}\n"
        )
        log_f.flush()

    train_loader, val_loader, class_counts = build_dataloaders(args)

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
            "WARNING: backbone is frozen but there is no --backbone_checkpoint — "
        )

    class_weights = None
    if args.use_class_weights and args.cb_loss_pow > 0:
        inv = (1.0 / class_counts.clamp(min=1)) ** args.cb_loss_pow
        class_weights = (inv / inv.sum() * args.num_classes).to(device)
        print("Loss class weights: " + ", ".join(f"{w:.3f}" for w in class_weights.tolist()))

    CE_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2, weight=class_weights).to(device)
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
    best_val_uar = 0.0   
    best_score = 0.0     

    if args.resume and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        best_val_uar = ckpt.get("best_val_uar", 0.0)
        best_score = ckpt.get("best_score", best_val_acc)
        if early_stopper and "es_counter" in ckpt:
            early_stopper.counter = ckpt["es_counter"]
            early_stopper.best = best_score
        print(f"  epoch={start_epoch}  best_score({args.select_metric})={best_score:.4f}")

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
        val_loss, val_acc, val_uar = validate_video(
            model, val_loader, CE_criterion,
            device, epoch, args.epochs,
            use_3dmm=args.use_3dmm,
            use_amp=args.amp,
            num_classes=args.num_classes,
        )

        val_score = {
            "war":  val_acc,
            "uar":  val_uar,
            "mean": 0.5 * (val_acc + val_uar),
        }[args.select_metric]
        is_best = val_score > best_score

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = (time.time() - t0) / 60.0
        es_count = early_stopper.counter if early_stopper else 0

        epoch_bar.set_postfix({
            "lr":    f"{lr:.1e}",
            "t_acc": f"{train_acc * 100:.1f}%",
            "WAR":   f"{val_acc * 100:.1f}%",
            "UAR":   f"{val_uar * 100:.1f}%",
            "best":  f"{best_score * 100:.1f}%",
            "ES":    f"{es_count}/{args.patience}" if args.patience > 0 else "off",
        })

        summary = (
            f"Ep {epoch + 1:3d}/{args.epochs} | "
            f"lr={lr:.2e} | "
            f"train {train_loss:.4f}/{train_acc * 100:.2f}% | "
            f"val {val_loss:.4f} WAR {val_acc * 100:.2f}% UAR {val_uar * 100:.2f}% | "
            f"{elapsed:.1f}min"
        )
        if is_best:
            summary += f"  ★ best ({args.select_metric})"
        tqdm.write(summary)

        if log_f is not None:
            log_f.write(
                f"{epoch + 1:^6d} {lr:^12.8f} {train_loss:^12.4f} {train_acc * 100:^10.2f} "
                f"{val_loss:^12.4f} {val_acc * 100:^10.2f} {val_uar * 100:^10.2f} {elapsed:^10.2f} "
                f"{es_count:^6d}\n"
            )
            log_f.flush()

        if is_best:
            best_score = val_score
            best_val_acc = val_acc
            best_val_uar = val_uar
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_uar": best_val_uar,
                "best_score": best_score,
                "select_metric": args.select_metric,
                "es_counter": es_count,
                "args": vars(args),
            }, resume_path)
            if log_f is not None:
                log_f.write(f"BEST\t{args.select_metric}={best_score * 100:.2f}\t"
                            f"WAR={best_val_acc * 100:.2f}\tUAR={best_val_uar * 100:.2f}\n")
                log_f.flush()

        if early_stopper and early_stopper.step(val_score):
            tqdm.write(
                f"\nEarly stopping after epoch {epoch + 1} "
                f"(no improvement for {args.patience} epochs)."
            )
            if log_f is not None:
                log_f.write(f"EARLY_STOP\tepoch={epoch + 1}\n")
            break

    epoch_bar.close()

    print(f"\nBest checkpoint ({args.select_metric}): "
          f"WAR={best_val_acc * 100:.2f}%  UAR={best_val_uar * 100:.2f}%")
    if log_f is not None:
        log_f.write(f"\nBest checkpoint ({args.select_metric}): "
                    f"WAR={best_val_acc * 100:.2f}%  UAR={best_val_uar * 100:.2f}%\n")
        log_f.close()


if __name__ == "__main__":
    main()
