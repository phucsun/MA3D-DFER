import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter

class CheoFaMo(Dataset):
    def __init__(self, root_dir, split="train", transform=None, test_size=0.2,
        seed=42, stats_path="cheofamo_stats.npz", verbose=False,
    ):
        self.root_dir = root_dir
        self.transform = transform

        # ---- load metadata ----
        with open(os.path.join(root_dir, "Metadata.json")) as f:
            meta = json.load(f)

        frames_dir = os.path.join(root_dir, "Frames")

        def is_valid(item):
            """Check validity of a sample."""
            if item["Offset_Index"] - item["Onset_Index"] < 4:
                return False
            if item["External Expression"] == "Other":
                return False
            return True

        def build_sample(item):
            """Build sample dictionary from metadata."""
            vid, cid, apex = item["Video_ID"], item["Character_ID"], item["Apex_Index"]
            folder = os.path.join(frames_dir, f"{int(vid):02d}_{int(cid):02d}_{int(apex):05d}")

            if not os.path.isdir(folder):
                return None

            imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            npys = [f for f in os.listdir(folder) if f.endswith(".npy")]

            if not imgs or not npys:
                return None

            return {
                "folder": folder,
                "img_path": os.path.join(folder, imgs[0]),
                "emotion": item["External Expression"]
            }

        # ---- build dataset ----
        samples, missing = [], 0

        for item in meta:
            if not is_valid(item):
                continue

            sample = build_sample(item)
            if sample is None:
                missing += 1
                continue

            samples.append(sample)

        # ---- label mapping ----
        emotions = sorted({s["emotion"] for s in samples})
        self.label2idx = {e: i for i, e in enumerate(emotions)}
        self.idx2label = {i: e for e, i in self.label2idx.items()}

        # ---- stratified split ----
        train, test = train_test_split(
            samples,
            test_size=test_size,
            stratify=[s["emotion"] for s in samples],
            random_state=seed
        )

        self.samples = train if split == "train" else test

        if verbose:
            print(f"[CheoFaMo] samples={len(samples)}, missing={missing}")
            print(f"[CheoFaMo] train={len(train)}, test={len(test)}")

        # ---- load stats ----
        self.stats = None
        stats_file = os.path.join(root_dir, stats_path)
        if os.path.exists(stats_file):
            raw = np.load(stats_file)
            self.stats = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # ---- image ----
        img = Image.open(s["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # ---- npy features ----
        feats = {}
        for f in os.listdir(s["folder"]):
            if not f.endswith(".npy"):
                continue

            key = f[:-4]
            x = torch.tensor(np.load(os.path.join(s["folder"], f)), dtype=torch.float32)

            # normalize if stats exist
            if self.stats and f"{key}_mean" in self.stats:
                x = (x - self.stats[f"{key}_mean"]) / self.stats[f"{key}_std"]

            feats[key] = x

        return {
            "image": img,
            "npy": feats,
            "label": torch.tensor(self.label2idx[s["emotion"]])
        }

def compute_cheofamo_stats(root_dir, output="cheofamo_stats.npz", strict_npy=None):
    frames_dir = os.path.join(root_dir, "Frames")

    with open(os.path.join(root_dir, "Metadata.json")) as f:
        meta = json.load(f)

    buffers = {}
    used = skipped = missing = 0

    def iter_folder(item):
        vid, cid, apex = item["Video_ID"], item["Character_ID"], item["Apex_Index"]
        return os.path.join(frames_dir, f"{int(vid):02d}_{int(cid):02d}_{int(apex):05d}")

    for item in meta:
        if item["Offset_Index"] - item["Onset_Index"] < 4:
            continue
        if item["External Expression"] == "Other":
            continue

        folder = iter_folder(item)
        if not os.path.isdir(folder):
            missing += 1
            continue

        npys = [f for f in os.listdir(folder) if f.endswith(".npy")]

        if strict_npy and len(npys) != strict_npy:
            skipped += 1
            continue
        if not npys:
            skipped += 1
            continue

        ok = True
        for f in npys:
            try:
                x = np.load(os.path.join(folder, f))
            except:
                ok = False
                break

            buffers.setdefault(f[:-4], []).append(x)

        used += int(ok)
        skipped += int(not ok)

    print(f"Used={used}, Skipped={skipped}, Missing={missing}")

    # ---- compute stats ----
    stats = {}
    for k, v in buffers.items():
        arr = np.stack(v)
        mean, std = arr.mean(0), arr.std(0)
        std[std < 1e-6] = 1.0

        stats[f"{k}_mean"] = mean
        stats[f"{k}_std"] = std

    np.savez(os.path.join(root_dir, output), **stats)
    print(f"Saved -> {output}")

if __name__ == "__main__":
    dataset = CheoFaMo(
        root_dir="Datasets/CheoFamo",
        split="train",
        verbose=True
    )