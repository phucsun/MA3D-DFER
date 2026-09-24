import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict


class CaersDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, stats_file="caers_stats.npz", verbose=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.data_dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.data_dir):
            raise RuntimeError(f"Missing directory: {self.data_dir}")

        # class mapping
        classes = sorted(d for d in os.listdir(self.data_dir)
                         if os.path.isdir(os.path.join(self.data_dir, d)))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        # load normalization stats
        self.stats = None
        stats_path = os.path.join(root_dir, stats_file)
        if os.path.exists(stats_path):
            raw = np.load(stats_path)
            self.stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}

        self.samples = self._scan_dataset(classes)
        self._log(verbose, len(classes))

    def _scan_dataset(self, classes):
        """Scan dataset once and build samples list."""
        samples = []

        for cls in classes:
            cls_path = os.path.join(self.data_dir, cls)
            label = self.class_to_idx[cls]

            for folder in os.listdir(cls_path):
                folder_path = os.path.join(cls_path, folder)
                if not os.path.isdir(folder_path):
                    continue

                files = os.listdir(folder_path)

                img = next((f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))), None)
                npys = [f for f in files if f.endswith(".npy")]

                if not img or not npys:
                    continue

                samples.append({
                    "folder": folder_path,
                    "img_path": os.path.join(folder_path, img),
                    "label": label
                })

        return samples

    def _log(self, verbose, n_classes):
        if verbose:
            print(f"[Caers Dataset - {self.split}]")
            print(f"  Samples : {len(self.samples)}")
            print(f"  Classes : {n_classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # ---- image ----
        image = Image.open(item["img_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # ---- load all npy features ----
        features = {}
        for f in os.listdir(item["folder"]):
            if not f.endswith(".npy"):
                continue

            key = f[:-4]
            x = torch.tensor(np.load(os.path.join(item["folder"], f)), dtype=torch.float32)

            # normalize if stats exist
            if self.stats and f"{key}_mean" in self.stats:
                x = (x - self.stats[f"{key}_mean"]) / self.stats[f"{key}_std"]

            features[key] = x

        return {
            "image": image,
            "npy": features,
            "label": torch.tensor(item["label"], dtype=torch.long)
        }

def compute_stats(root_dir, split="train", output="caers_stats.npz"):
    data_dir = os.path.join(root_dir, split)
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"Missing directory: {data_dir}")

    buffers = defaultdict(list)
    skipped = 0

    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)

        for folder in os.listdir(cls_path):
            folder_path = os.path.join(cls_path, folder)
            if not os.path.isdir(folder_path):
                continue

            npys = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
            if not npys:
                skipped += 1
                continue

            for f in npys:
                try:
                    buffers[f[:-4]].append(np.load(os.path.join(folder_path, f)))
                except:
                    skipped += 1

    print(f"Skipped: {skipped}")

    stats = {}
    for k, v in buffers.items():
        arr = np.stack(v)
        mean, std = arr.mean(0), arr.std(0)
        std[std < 1e-6] = 1.0

        stats[f"{k}_mean"] = mean
        stats[f"{k}_std"] = std

        print(f"{k}: {mean.shape}")

    np.savez(os.path.join(root_dir, output), **stats)
    print(f"Saved: {output}")

if __name__ == "__main__":
    compute_stats("Datasets/caers/caers", "train")

    dataset = CaersDataset(
        root_dir="Datasets/caers/caers",
        split="train",
        verbose=True
    )