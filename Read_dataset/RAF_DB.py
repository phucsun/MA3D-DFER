import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict


class RAFDataset(Dataset):
    def __init__(
            self,
            root_dir,
            is_train=True,
            transform=None,
            stats_path=None,
            verbose=False,
    ):
        self.transform = transform
        self.samples = []

        split_dir = "train" if is_train else "test"
        self.data_dir = os.path.join(root_dir, split_dir)

        if not os.path.exists(self.data_dir):
            raise RuntimeError(f"Directory not found: {self.data_dir}")

        # Load normalization stats
        self.stats = None
        if stats_path is not None:
            stats = np.load(os.path.join(root_dir, stats_path))
            self.stats = {
                k: torch.from_numpy(v).float()
                for k, v in stats.items()
            }

        skipped = 0

        for label_folder in sorted(os.listdir(self.data_dir)):
            label_path = os.path.join(self.data_dir, label_folder)
            if not os.path.isdir(label_path):
                continue

            label = int(label_folder) - 1

            for sample_folder in sorted(os.listdir(label_path)):
                sample_path = os.path.join(label_path, sample_folder)
                if not os.path.isdir(sample_path):
                    continue

                # img_path = os.path.join(sample_path, "inputs.png")

                img_name = sample_folder + ".jpg"
                img_path = os.path.join(sample_path, img_name)

                if not os.path.exists(img_path):
                    skipped += 1
                    continue

                npy_files = [
                    f for f in os.listdir(sample_path)
                    if f.endswith(".npy") and not f.endswith("_lmk.npy")
                ]

                if len(npy_files) < 5:
                    skipped += 1
                    continue

                self.samples.append(
                    {
                        "folder": sample_path,
                        "img_path": img_path,
                        "label": label,
                    }
                )
        if verbose:
            print(
                f"[RAFDataset] {split_dir}: "
                f"{len(self.samples)} samples loaded "
            )
            print("[DEBUG] First 5 samples:")
            for i in range(min(5, len(self.samples))):
                print(self.samples[i]["img_path"])

            counter = Counter([s["label"] for s in self.samples])
            print("[DEBUG] Label distribution after stratified subset:")
            for k in sorted(counter):
                print(f"  Label {k}: {counter[k]}")

        self.expected_keys = ["shape", "tex", "exp", "pose", "detail"]
        self.n_shape = 100
        self.n_tex = 50
        self.n_exp = 50
        self.n_pose = 6
        self.n_detail = 128

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Image
        image = Image.open(sample["img_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load + normalize ALL npy
        npy_norm = {}

        for key in self.expected_keys:
            path = os.path.join(sample["folder"], f"{key}.npy")

            if os.path.exists(path):
                x = torch.from_numpy(np.load(path)).float()
            else:
                # Fill missing with zeros
                if key == "shape":
                    x = torch.zeros(self.n_shape)
                elif key == "tex":
                    x = torch.zeros(self.n_tex)
                elif key == "exp":
                    x = torch.zeros(self.n_exp)
                elif key == "pose":
                    x = torch.zeros(self.n_pose)
                elif key == "detail":
                    x = torch.zeros(self.n_detail)

            # normalize if stats exist
            if self.stats is not None:
                mean_key = f"{key}_mean"
                std_key = f"{key}_std"
                if mean_key in self.stats and std_key in self.stats:
                    x = (x - self.stats[mean_key]) / self.stats[std_key]

            npy_norm[key] = x

        label = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "image": image,
            "npy": npy_norm,
            "label": label,
        }


# =========================================================
# Compute dataset statistics
# =========================================================
def compute_stats(root_dir, split="train",
                  output_name="rafdb_emoca_stats.npz",
                  skip_landmark=True,
                  min_samples=10):

    split_dir = os.path.join(root_dir, split)
    if not os.path.exists(split_dir):
        raise RuntimeError(f"Missing split: {split_dir}")

    buffers = defaultdict(list)

    print(f"[Stats] scanning {split}")

    for label_dir in os.listdir(split_dir):
        label_path = os.path.join(split_dir, label_dir)
        if not os.path.isdir(label_path):
            continue

        for sample_dir in os.listdir(label_path):
            path = os.path.join(label_path, sample_dir)
            if not os.path.isdir(path):
                continue

            npy_files = [f for f in os.listdir(path) if f.endswith(".npy")]

            valid = False
            for f in npy_files:
                if skip_landmark and f.endswith("_lmk.npy"):
                    continue

                key = f[:-4]
                x = np.load(os.path.join(path, f)).reshape(-1)

                buffers[key].append(x)
                valid = True

            if not valid:
                continue

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

        print(f"{key}: {data.shape}")

    out_path = os.path.join(root_dir, output_name)
    np.savez(out_path, **stats)

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    compute_stats(
        root_dir="Datasets/RAF-DB_lmk",
        split="train",
        output_name="rafdb_emoca_stats.npz"
    )