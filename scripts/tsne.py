import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.manifold import TSNE

from models.MA3D import MA3D
from build_dataloader import get_dataloaders
from engine import prepare_batch


EMOTION_NAMES = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default="RAF-DB")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/last.pth")
    return parser.parse_args()


def load_model(ckpt_path, device, num_classes):
    model = MA3D(num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def extract_features(model, loader, device):
    feats, labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            images, y, x3d = prepare_batch(batch, device)
            _, feat, _ = model(images, x3d)

            feats.append(feat.cpu().numpy())
            labels.append(y.cpu().numpy())

    return np.concatenate(feats), np.concatenate(labels)


def visualize_tsne(features, labels, save_path):
    print("Running t-SNE...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=42
    )

    z = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 8))

    num_classes = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", num_classes)

    for c in range(num_classes):
        idx = labels == c
        ax.scatter(z[idx, 0], z[idx, 1],
                   s=25, alpha=0.85,
                   color=cmap(c),
                   label=EMOTION_NAMES.get(c, str(c)))

    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max * 1.25)

    ax.legend(loc="upper right",
              frameon=True,
              fontsize=10,
              framealpha=0.9,
              borderpad=0.6,
              edgecolor="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved to {save_path}")


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_loader = get_dataloaders(args)
    model = load_model(args.ckpt_path, device, args.num_classes)

    features, labels = extract_features(model, val_loader, device)

    base = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    save_path = f"tsne_{base}.png"

    visualize_tsne(features, labels, save_path)


if __name__ == "__main__":
    main()