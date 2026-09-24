import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm

from models.MA3D import MA3D
from build_dataloader import get_dataloaders
from engine import prepare_batch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default="RAF-DB")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/last.pth")
    return parser.parse_args()

def load_model(ckpt_path, device, num_classes):
    model = MA3D(num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model

def denormalize(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = x.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)


def save_images(img_tensor, attn, save_dir, idx):
    img = denormalize(img_tensor)
    img = cv2.resize(img, (224, 224))
    h, w, _ = img.shape

    # Save original image
    orig = (img * 255).astype(np.uint8)
    cv2.imwrite(
        os.path.join(save_dir, f"{idx}_orig.png"),
        cv2.cvtColor(orig, cv2.COLOR_RGB2BGR),
    )

    # Process attention map
    attn = attn.mean(0)[0, 1:]
    attn = attn.reshape(7, 7).cpu().numpy()
    attn = (attn - attn.min()) / (attn.max() + 1e-8)
    attn = cv2.resize(attn, (w, h))

    heatmap = cv2.applyColorMap((attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = heatmap[:, :, ::-1] / 255.0

    # Overlay
    overlay = np.clip(0.6 * img + 0.4 * heatmap, 0, 1)

    cv2.imwrite(
        os.path.join(save_dir, f"{idx}_attn.png"),
        cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )

def extract_features(model, loader, device, save_dir="attn_vis"):
    os.makedirs(save_dir, exist_ok=True)

    target_ids = {142, 136, 297, 358, 352, 364}

    with torch.no_grad():
        for b, batch in enumerate(tqdm(loader, desc="Processing")):
            images, labels, x3d = prepare_batch(batch, device)
            start = b * loader.batch_size

            idx = [i for i in range(len(images)) if start + i in target_ids]
            if not idx:
                continue

            images_sel = images[idx]
            x3d_sel = x3d[idx]

            _, _, attns = model(images_sel, x3d_sel)
            attn = attns[-1]["s"]["img_attn"]

            for j, i in enumerate(idx):
                save_images(
                    images_sel[j].cpu(),
                    attn[j].cpu(),
                    save_dir,
                    start + i,
                )

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_loader = get_dataloaders(args)
    model = load_model(args.ckpt_path, device, args.num_classes)

    extract_features(model, val_loader, device, save_dir="results/attn_vis_MA_2")


if __name__ == "__main__":
    main()