import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FERPlusDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, stats_path="ferplus_stats.npz", verbose=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.data_dir = os.path.join(root_dir, split)
        if not os.path.exists(self.data_dir):
            raise RuntimeError(f"Directory not found: {self.data_dir}")

        # ---- class mapping ----
        class_names = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        # ---- load normalization stats ----
        self.stats = None
        stats_file = os.path.join(root_dir, stats_path)
        if os.path.exists(stats_file):
            stats = np.load(stats_file)
            self.stats = {k: torch.from_numpy(v).float() for k, v in stats.items()}

        # ---- build dataset index ----
        self.samples = []
        skipped = 0

        for cls in class_names:
            cls_path = os.path.join(self.data_dir, cls)
            label = self.class_to_idx[cls]

            for folder in os.listdir(cls_path):
                folder_path = os.path.join(cls_path, folder)
                if not os.path.isdir(folder_path):
                    continue

                files = os.listdir(folder_path)

                img_files = [f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                npy_files = [f for f in files if f.endswith(".npy")]

                if not img_files or not npy_files:
                    skipped += 1
                    continue

                self.samples.append({
                    "folder": folder_path,
                    "img_path": os.path.join(folder_path, img_files[0]),
                    "label": label
                })

        if verbose:
            print(f"[FERPlusDataset] split={split}")
            print(f"  samples={len(self.samples)} | classes={len(class_names)} | skipped={skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ---- load image ----
        image = Image.open(sample["img_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # ---- load npy features ----
        features = {}
        for f in os.listdir(sample["folder"]):
            if not f.endswith(".npy"):
                continue

            key = f[:-4]
            x = torch.from_numpy(np.load(os.path.join(sample["folder"], f))).float()

            # normalize if stats exist
            if self.stats and f"{key}_mean" in self.stats:
                x = (x - self.stats[f"{key}_mean"]) / self.stats[f"{key}_std"]

            features[key] = x

        return {
            "image": image,
            "npy": features,
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


def compute_ferplus_stats(root_dir, split="train", output="ferplus_stats.npz", expected_npy_count=None):
    """Compute mean and std for all npy features in dataset."""

    data_dir = os.path.join(root_dir, split)
    if not os.path.exists(data_dir):
        raise RuntimeError(f"Directory not found: {data_dir}")

    buffers = {}
    used = skipped = 0

    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)

        for folder in os.listdir(cls_path):
            folder_path = os.path.join(cls_path, folder)
            if not os.path.isdir(folder_path):
                continue

            npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

            if not npy_files or (expected_npy_count and len(npy_files) != expected_npy_count):
                skipped += 1
                continue

            for f in npy_files:
                key = f[:-4]
                try:
                    arr = np.load(os.path.join(folder_path, f))
                except Exception:
                    skipped += 1
                    continue

                buffers.setdefault(key, []).append(arr)

            used += 1

    print(f"Used: {used} | Skipped: {skipped}")

    stats = {}
    for key, values in buffers.items():
        data = np.stack(values)
        mean, std = data.mean(0), data.std(0)

        std = np.where(std < 1e-6, 1.0, std)

        stats[f"{key}_mean"] = mean
        stats[f"{key}_std"] = std

        print(f"{key}: shape={mean.shape}")

    out_path = os.path.join(root_dir, output)
    np.savez(out_path, **stats)

    print(f"Saved stats to: {out_path}")