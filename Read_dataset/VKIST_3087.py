import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

class VKIST_Dataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None,
                 stats_path="micro_sb_emoca_stats.npz", verbose=False):

        self.transform = transform
        split = "train" if is_train else "test"
        self.data_dir = os.path.join(root_dir, split)

        if not os.path.exists(self.data_dir):
            raise RuntimeError(f"Missing directory: {self.data_dir}")

        # ---- load normalization stats ----
        self.stats = None
        if stats_path:
            stats = np.load(stats_path)
            self.stats = {k: torch.from_numpy(v).float() for k, v in stats.items()}

        self.samples = []
        skipped = 0

        # ---- scan dataset ----
        for folder in sorted(os.listdir(self.data_dir)):
            path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(path):
                continue

            # extract label from folder name (e.g., E01_xxx)
            label = next((int(p[1:]) - 1 for p in folder.split("_") if p.startswith("E")), None)
            if label is None:
                skipped += 1
                continue

            img_path = os.path.join(path, "inputs.png")
            if not os.path.exists(img_path):
                skipped += 1
                continue

            self.samples.append({
                "folder": path,
                "img_path": img_path,
                "label": label
            })

        if verbose:
            print(f"[VKIST] samples={len(self.samples)} skipped={skipped}")

    def __len__(self):
        return len(self.samples)

    def _load_npy(self, folder, key):
        """Load npy and apply normalization if stats exist."""
        path = os.path.join(folder, f"{key}.npy")

        if os.path.exists(path):
            x = torch.from_numpy(np.load(path)).float()
        else:
            return None

        if self.stats and f"{key}_mean" in self.stats:
            x = (x - self.stats[f"{key}_mean"]) / self.stats[f"{key}_std"]

        return x

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ---- image ----
        img = Image.open(sample["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # ---- load all npy features ----
        features = {}
        for f in os.listdir(sample["folder"]):
            if f.endswith(".npy") and f != "inputs_lmk.npy":
                key = f[:-4]
                x = self._load_npy(sample["folder"], key)
                if x is not None:
                    features[key] = x

        return {
            "image": img,
            "npy": features,
            "label": torch.tensor(sample["label"])
        }

def compute_vkist_stats(root_dir, split="train",
                        output_name="micro_sb_emoca_stats.npz",
                        min_samples=10):

    split_dir = os.path.join(root_dir, split)
    if not os.path.exists(split_dir):
        raise RuntimeError(f"Missing split: {split_dir}")

    buffers = defaultdict(list)

    print(f"[Stats] scanning {split}")

    # ---- collect data ----
    for folder in os.listdir(split_dir):
        path = os.path.join(split_dir, folder)
        if not os.path.isdir(path):
            continue

        if not os.path.exists(os.path.join(path, "inputs.png")):
            continue

        for f in os.listdir(path):
            if not f.endswith(".npy"):
                continue

            key = f[:-4]
            x = np.load(os.path.join(path, f)).reshape(-1)
            buffers[key].append(x)

    # ---- compute stats ----
    stats = {}

    for key, values in buffers.items():
        if len(values) < min_samples:
            continue

        data = np.stack(values)
        mean, std = data.mean(0), data.std(0)

        std = np.where(std < 1e-6, 1.0, std)

        stats[f"{key}_mean"] = mean
        stats[f"{key}_std"] = std

        print(f"{key}: N={data.shape[0]} D={data.shape[1]}")

    out_path = os.path.join(root_dir, output_name)
    np.savez(out_path, **stats)

    print(f"\nSaved stats -> {out_path}")

if __name__ == "__main__":
    compute_vkist_stats(
        root_dir="Datasets/3087data_subject_split",
        split="train",
        output_name="micro_sb_emoca_stats.npz"
    )