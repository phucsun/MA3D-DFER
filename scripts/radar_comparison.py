import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from models.MA3D import MA3D
from build_dataloader import get_dataloaders
from engine import prepare_batch


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["save", "vis"], default="save")

    parser.add_argument("--data_type", default="RAF-DB")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/last.pth")
    parser.add_argument("--out_name", type=str, default="baseline")

    parser.add_argument("--baseline_path", type=str, default="results/baseline.npy")
    parser.add_argument("--ours_path", type=str, default="results/ours.npy")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="results")

    return parser.parse_args()

def draw_radar(baseline, ours, labels, save_path):
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    base = list(baseline) + [baseline[0]]
    ours = list(ours) + [ours[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, base, marker="s", linewidth=2, label="Baseline")
    ax.fill(angles, base, alpha=0.1)

    ax.plot(angles, ours, marker="o", linewidth=2, label="Ours")
    ax.fill(angles, ours, alpha=0.2)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    ax.set_ylim(0, 100)
    ax.set_rticks([20, 40, 60, 80])
    ax.set_rlabel_position(180)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.title("Class-wise Accuracy + Overall Accuracy", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def extract_results(args, device):
    _, val_loader = get_dataloaders(args)

    model = MA3D(num_classes=args.num_classes).to(device)
    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state.get("model", state))
    model.eval()

    preds_all, labels_all = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            x, y, x3d = prepare_batch(batch, device)
            logits, _, _ = model(x, x3d)

            preds_all.extend(torch.argmax(logits, 1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    cm = confusion_matrix(labels_all, preds_all, labels=range(args.num_classes))

    class_acc = (cm.diagonal() / (cm.sum(1) + 1e-8)) * 100
    acc = (np.sum(np.diag(cm)) / (np.sum(cm) + 1e-8)) * 100

    results = np.append(class_acc, acc)

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"{args.out_name}.npy")
    np.save(path, results)

    print(f"Saved: {path}")
    print(f"Accuracy: {acc:.2f}%")

def visualize(args):
    if not (os.path.exists(args.baseline_path) and os.path.exists(args.ours_path)):
        print("Missing result files")
        return

    base = np.load(args.baseline_path)
    ours = np.load(args.ours_path)

    n_cls = len(base) - 1
    labels = [f"C{i}" for i in range(n_cls)] + ["Acc"]

    save_path = os.path.join(args.save_dir, "radar_comparison.png")

    draw_radar(base, ours, labels, save_path)

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "save":
        extract_results(args, device)
    else:
        visualize(args)


if __name__ == "__main__":
    main()