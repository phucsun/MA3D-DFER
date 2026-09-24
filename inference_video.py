"""
inference_video.py — evaluate an MA3D-Video checkpoint on the validation set.

Runs a checkpoint (default checkpoints/video_best.pth) on the validation
split and prints:
    - Accuracy per emotion (= per-class recall)
    - Confusion matrix
    - F1-score (per-class + macro + weighted)
    - UAR (Unweighted Average Recall) = mean of per-class recall
    - WAR (Weighted Average Recall)   = overall accuracy

UAR/WAR follow the DFER convention from S2D (2312.05447v2.pdf):
    WAR = weighted average recall = correct predictions / total samples.
    UAR = unweighted average recall = mean of per-class recall (class-balanced).

Model config (use_3dmm, temporal_module, model_type, hidden_dim, max_frames,
frame_step, num_classes...) is read AUTOMATICALLY from the args stored in the
checkpoint so it matches training; it can be overridden via the CLI.

Example:
    python inference_video.py
    python inference_video.py --checkpoint checkpoints/video_best.pth \
        --data_dir dataset/CAER/caer_3dmm --val_split validation \
        --stats_path dataset/CAER/caer_3dmm/video_3dmm_stats.npz
"""

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
)

from models.MA3D_Video import MA3D_Video
from Read_dataset import VideoDataset, collate_video_fn


def get_args():
    p = argparse.ArgumentParser("MA3D-Video Inference / Evaluation")
    p.add_argument("--checkpoint", type=str, default="checkpoints/video_best.pth",
                   help="Checkpoint path (.pth)")
    p.add_argument("--data_dir", type=str, default="dataset/CAER/caer_3dmm",
                   help="Dataset root directory containing the validation/ subfolder")
    p.add_argument("--val_split", type=str, default="validation",
                   help="Validation subfolder name")
    p.add_argument("--stats_path", type=str,
                   default="dataset/CAER/caer_3dmm/video_3dmm_stats.npz",
                   help="3DMM mean/std .npz file (only needed when use_3dmm=True)")

    # The arguments below default to None -> read from checkpoint['args'].
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--model_type", type=str, default=None,
                   choices=["small", "base", "large"])
    p.add_argument("--temporal_module", type=str, default=None,
                   choices=["lstm", "transformer", "mean"])
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--head_dropout", type=float, default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--frame_step", type=int, default=None)
    p.add_argument("--use_3dmm", dest="use_3dmm", action="store_true", default=None)
    p.add_argument("--no_3dmm", dest="use_3dmm", action="store_false")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_amp", dest="amp", action="store_false", default=True)
    p.add_argument("--save_pdf", type=str, default="inference_report.pdf",
                   help="Save the full classification report to a PDF file "
                        "(per-emotion metrics, confusion matrix, overall metrics). "
                        "Set to '' to disable.")
    p.add_argument("--save_csv", type=str, default=None,
                   help="(optional) also save the confusion matrix to a CSV file")
    return p.parse_args()


def resolve_cfg(args, ckpt_args):
    """Use the CLI value if set, otherwise fall back to the checkpoint args."""
    def pick(cli_val, key, fallback):
        if cli_val is not None:
            return cli_val
        if ckpt_args is not None and key in ckpt_args and ckpt_args[key] is not None:
            return ckpt_args[key]
        return fallback

    return {
        "num_classes":     pick(args.num_classes,     "num_classes",     7),
        "model_type":      pick(args.model_type,      "model_type",      "large"),
        "temporal_module": pick(args.temporal_module, "temporal_module", "lstm"),
        "hidden_dim":      pick(args.hidden_dim,       "hidden_dim",      512),
        "head_dropout":    pick(args.head_dropout,     "head_dropout",    0.3),
        "max_frames":      pick(args.max_frames,       "max_frames",      16),
        "frame_step":      pick(args.frame_step,       "frame_step",      2),
        "use_3dmm":        pick(args.use_3dmm,         "use_3dmm",        True),
    }


def print_table(rows, headers):
    widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        print("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))


def save_report_pdf(pdf_path, class_names, cm, per_class_recall, per_class_prec,
                    per_class_f1, support, uar, war, macro_f1, weighted_f1,
                    report_text, meta):
    """Export the full evaluation report to a multi-page PDF file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n = len(class_names)

    with PdfPages(pdf_path) as pdf:
        # ---- Page 1: summary + per-emotion table ----
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.suptitle("MA3D-Video — Evaluation Report", fontsize=16, fontweight="bold")

        # Metadata + overall metrics block (text)
        ax_txt = fig.add_axes([0.07, 0.70, 0.86, 0.20])
        ax_txt.axis("off")
        info = (
            f"Checkpoint : {meta.get('checkpoint')}\n"
            f"Epoch      : {meta.get('epoch')}      Samples: {meta.get('n_samples')}\n"
            f"Dataset    : {meta.get('data_dir')}  [{meta.get('val_split')}]\n"
            f"Config     : type={meta.get('model_type')}  temporal={meta.get('temporal_module')}  "
            f"use_3dmm={meta.get('use_3dmm')}  max_frames={meta.get('max_frames')}  "
            f"frame_step={meta.get('frame_step')}\n\n"
            f"WAR (Weighted Avg Recall / Accuracy) : {war * 100:.2f}%\n"
            f"UAR (Unweighted Avg Recall)          : {uar * 100:.2f}%\n"
            f"Macro F1-score                       : {macro_f1 * 100:.2f}%\n"
            f"Weighted F1-score                    : {weighted_f1 * 100:.2f}%"
        )
        ax_txt.text(0.0, 1.0, info, va="top", ha="left", family="monospace", fontsize=9.5)

        # Per-emotion table
        ax_tbl = fig.add_axes([0.07, 0.30, 0.86, 0.34])
        ax_tbl.axis("off")
        ax_tbl.set_title("Per-Emotion Metrics", fontsize=12, fontweight="bold", pad=8)
        col_labels = ["Emotion", "Support", "Acc/Recall%", "Precision%", "F1%"]
        cell_text = []
        for i, name in enumerate(class_names):
            cell_text.append([
                name, str(int(support[i])),
                f"{per_class_recall[i] * 100:.2f}",
                f"{per_class_prec[i] * 100:.2f}",
                f"{per_class_f1[i] * 100:.2f}",
            ])
        cell_text.append([
            "OVERALL (UAR/–/–)", str(int(support.sum())),
            f"{uar * 100:.2f}", f"{war * 100:.2f}", f"{macro_f1 * 100:.2f}",
        ])
        tbl = ax_tbl.table(cellText=cell_text, colLabels=col_labels,
                           loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.5)
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#40466e")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for j in range(len(col_labels)):
            tbl[len(cell_text), j].set_facecolor("#d9e1f2")
            tbl[len(cell_text), j].set_text_props(fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2: confusion matrices (counts + row-normalized) ----
        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
        fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

        def _heatmap(ax, mat, title, fmt):
            im = ax.imshow(mat, cmap="Blues", aspect="auto")
            ax.set_title(title, fontsize=11, pad=8)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            thresh = mat.max() / 2.0 if mat.max() > 0 else 0.5
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                            color="white" if mat[i, j] > thresh else "black", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        _heatmap(axes[0], cm, "Counts (rows = true, cols = pred)", "d")
        _heatmap(axes[1], cm_norm, "Row-normalized (recall per class)", ".2f")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 3: sklearn classification_report (monospace) ----
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("sklearn classification_report", fontsize=14, fontweight="bold")
        ax = fig.add_axes([0.07, 0.05, 0.86, 0.86])
        ax.axis("off")
        ax.text(0.0, 1.0, report_text, va="top", ha="left",
                family="monospace", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nPDF report saved to {pdf_path}")


@torch.no_grad()
def evaluate(model, loader, device, use_3dmm, use_amp):
    model.eval()
    all_preds, all_labels = [], []
    is_cuda = device != "cpu" and torch.cuda.is_available()

    for batch in tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True):
        video = batch["video"].to(device, non_blocking=True)
        labels = batch["labels"]
        lengths = batch["lengths"].to(device, non_blocking=True)
        video_3d = None
        if use_3dmm and "video_3d" in batch:
            video_3d = batch["video_3d"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp and is_cuda):
            logits, _ = model(video, video_3d, seq_lengths=lengths)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", None)
    cfg = resolve_cfg(args, ckpt_args)

    print(f"\nCheckpoint : {args.checkpoint}")
    print(f"  epoch        = {ckpt.get('epoch')}")
    print(f"  best_val_acc = {ckpt.get('best_val_acc')}")
    print(f"Config (resolved):")
    for k, v in cfg.items():
        print(f"  {k:<16}= {v}")

    # ----- Dataset / Loader -----
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    stats_path = args.stats_path if cfg["use_3dmm"] else None
    if cfg["use_3dmm"] and (not stats_path or not os.path.exists(stats_path)):
        print(f"WARNING: use_3dmm=True but stats_path does not exist ({stats_path}). "
              f"3DMM will NOT be normalized.")

    val_dataset = VideoDataset(
        root=args.data_dir,
        split=args.val_split,
        transform=val_transform,
        use_3dmm=cfg["use_3dmm"],
        max_frames=cfg["max_frames"],
        frame_step=cfg["frame_step"],
        stats_path=stats_path,
    )
    class_names = val_dataset.class_names
    print(f"\nValidation: {len(val_dataset)} samples | "
          f"mode={val_dataset.mode} | classes={class_names}")

    is_windows = os.name == "nt"
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0 and not is_windows),
        collate_fn=collate_video_fn,
    )

    # ----- Model -----
    model = MA3D_Video(
        num_classes=cfg["num_classes"],
        type=cfg["model_type"],
        use_3dmm=cfg["use_3dmm"],
        temporal_module=cfg["temporal_module"],
        hidden_dim=cfg["hidden_dim"],
        freeze_backbone=False,
        head_dropout=cfg["head_dropout"],
    ).to(device)

    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  missing (first few): {missing[:5]}")
        if unexpected:
            print(f"  unexpected (first few): {unexpected[:5]}")

    # ----- Evaluate -----
    y_true, y_pred = evaluate(model, val_loader, device, cfg["use_3dmm"], args.amp)

    labels_idx = list(range(cfg["num_classes"]))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    # Per-class recall = accuracy per emotion
    per_class_recall = recall_score(y_true, y_pred, labels=labels_idx,
                                    average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, labels=labels_idx,
                                     average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels_idx,
                            average=None, zero_division=0)
    support = cm.sum(axis=1)

    macro_f1 = f1_score(y_true, y_pred, labels=labels_idx, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels_idx, average="weighted", zero_division=0)

    uar = recall_score(y_true, y_pred, labels=labels_idx, average="macro", zero_division=0)
    war = (y_true == y_pred).mean()  # = accuracy = weighted average recall

    # ----- Report -----
    print("\n" + "=" * 70)
    print("PER-EMOTION METRICS")
    print("=" * 70)
    rows = []
    for i, name in enumerate(class_names):
        rows.append([
            name,
            int(support[i]),
            f"{per_class_recall[i] * 100:.2f}",
            f"{per_class_prec[i] * 100:.2f}",
            f"{per_class_f1[i] * 100:.2f}",
        ])
    print_table(rows, ["Emotion", "Support", "Acc/Recall%", "Precision%", "F1%"])

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX  (rows = true, cols = pred)")
    print("=" * 70)
    short = [n[:6] for n in class_names]
    header = "true\\pred".ljust(10) + "".join(s.rjust(8) for s in short)
    print(header)
    for i, name in enumerate(class_names):
        line = name[:9].ljust(10) + "".join(str(int(cm[i, j])).rjust(8)
                                             for j in range(len(class_names)))
        print(line)

    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    print(f"  WAR (Weighted Avg Recall / Accuracy) : {war * 100:.2f}%")
    print(f"  UAR (Unweighted Avg Recall)          : {uar * 100:.2f}%")
    print(f"  Macro F1-score                       : {macro_f1 * 100:.2f}%")
    print(f"  Weighted F1-score                    : {weighted_f1 * 100:.2f}%")
    print(f"  Total samples                        : {len(y_true)}")

    report_text = classification_report(y_true, y_pred, labels=labels_idx,
                                         target_names=class_names, digits=4,
                                         zero_division=0)
    print("\n--- sklearn classification_report ---")
    print(report_text)

    if args.save_pdf:
        meta = {
            "checkpoint": args.checkpoint,
            "epoch": ckpt.get("epoch"),
            "n_samples": len(y_true),
            "data_dir": args.data_dir,
            "val_split": args.val_split,
            "model_type": cfg["model_type"],
            "temporal_module": cfg["temporal_module"],
            "use_3dmm": cfg["use_3dmm"],
            "max_frames": cfg["max_frames"],
            "frame_step": cfg["frame_step"],
        }
        save_report_pdf(
            args.save_pdf, class_names, cm,
            per_class_recall, per_class_prec, per_class_f1, support,
            uar, war, macro_f1, weighted_f1, report_text, meta,
        )

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + class_names)
            for i, name in enumerate(class_names):
                w.writerow([name] + [int(cm[i, j]) for j in range(len(class_names))])
        print(f"Confusion matrix saved to {args.save_csv}")


if __name__ == "__main__":
    main()
