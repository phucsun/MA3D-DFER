import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

EMOTIONS = [
    "Surprise", "Fear", "Disgust",
    "Happiness", "Sadness", "Anger", "Neutral"
]
EMO2IDX = {e: i for i, e in enumerate(EMOTIONS)}

class DatasetCheo(Dataset):
    def __init__(self, json_path, root_dir, transform=None,
        stats_path="cheo_emoca_stats.npz", skip_missing=True, verbose=False,
    ):
        self.root_dir = root_dir
        self.transform = transform

        # ---- load normalization stats ----
        stats = np.load(os.path.join(root_dir, stats_path))
        self.stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in stats.items()}

        # ---- load annotations ----
        with open(os.path.join(root_dir, json_path), "r", encoding="utf-8") as f:
            data = json.load(f)

        def build_folder(item):
            """Construct sample folder path."""
            return os.path.join(
                root_dir,
                item["trich_doan"],
                f"person_{item['person_id']}_frame_{item['apex_frame_id']}"
            )

        self.samples = []
        skip_img = skip_npy = 0

        for item in data:
            emotion = item.get("external_vote")
            if emotion not in EMO2IDX:
                continue

            folder = build_folder(item)
            img_path = os.path.join(folder, "inputs.png")

            # ---- basic validity checks ----
            if not os.path.isdir(folder):
                skip_npy += 1
                if skip_missing:
                    continue

            if not os.path.exists(img_path):
                skip_img += 1
                if skip_missing:
                    continue

            npys = [f for f in os.listdir(folder) if f.endswith(".npy")]
            if len(npys) != 5:
                skip_npy += 1
                if skip_missing:
                    continue

            self.samples.append({
                "folder": folder,
                "img_path": img_path,
                "label": EMO2IDX[emotion]
            })

        if verbose:
            print(f"[DatasetCheo] samples={len(self.samples)}")
            print(f"  missing image : {skip_img}")
            print(f"  invalid npy   : {skip_npy}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # ---- image ----
        img = Image.open(s["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # ---- load + normalize npy ----
        feats = {}
        for f in os.listdir(s["folder"]):
            if not f.endswith(".npy"):
                continue

            key = f[:-4]
            x = torch.tensor(np.load(os.path.join(s["folder"], f)), dtype=torch.float32)

            mean_key, std_key = f"{key}_mean", f"{key}_std"
            if mean_key in self.stats and std_key in self.stats:
                x = (x - self.stats[mean_key]) / self.stats[std_key]

            feats[key] = x

        return {
            "image": img,
            "npy": feats,
            "label": torch.tensor(s["label"])
        }

def compute_cheo_stats(
    json_path,
    root_dir,
    output="cheo_emoca_stats.npz",
    expected_npy=5,
):
    with open(os.path.join(root_dir, json_path), "r", encoding="utf-8") as f:
        data = json.load(f)

    buffers = {}
    used = skipped = 0

    def folder_of(item):
        return os.path.join(
            root_dir,
            item["trich_doan"],
            f"person_{item['person_id']}_frame_{item['apex_frame_id']}"
        )

    for item in tqdm(data, desc="Processing"):
        folder = folder_of(item)

        if not os.path.isdir(folder):
            skipped += 1
            continue

        npys = [f for f in os.listdir(folder) if f.endswith(".npy")]
        if len(npys) != expected_npy:
            skipped += 1
            continue

        for f in npys:
            key = f[:-4]
            try:
                x = np.load(os.path.join(folder, f))
            except:
                skipped += 1
                continue

            buffers.setdefault(key, []).append(x)

        used += 1

    print(f"Used={used}, Skipped={skipped}")

    # ---- compute stats ----
    stats = {}
    for k, v in buffers.items():
        arr = np.stack(v)
        mean, std = arr.mean(0), arr.std(0)
        std[std < 1e-6] = 1.0

        stats[f"{k}_mean"] = mean
        stats[f"{k}_std"] = std

        print(f"{k}: {mean.shape}")

    out_path = os.path.join(root_dir, output)
    np.savez(out_path, **stats)
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    compute_cheo_stats(
        json_path="votes_train.json",
        root_dir="Datasets/cheo_dataset"
    )

    dataset = DatasetCheo(
        json_path="votes_train.json",
        root_dir="Datasets/cheo_dataset",
        verbose=True
    )