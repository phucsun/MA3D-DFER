"""
VideoDataset — dataset for video facial expression recognition.

Two modes are auto-detected from the directory layout:

  Mode A — video-file (e.g. CAER):
    root/split/ClassName/0001.avi
    root/split/ClassName/0002.avi

  Mode B — frame-folder (pre-extracted frames):
    root/split/ClassName/video_001/frame_001.jpg
    root/split/ClassName/video_001/frame_002.jpg
    root/split/ClassName/video_001/frame_001_shape.npy  # 3DMM optional

When use_3dmm=False, the .npy files are not required.
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

_VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv", ".webm")
_IMG_EXTENSIONS   = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_3DMM_KEYS = ["shape", "tex", "exp", "pose", "detail"]
_3DMM_DIMS = {"shape": 100, "tex": 50, "exp": 50, "pose": 6, "detail": 128}
_3DMM_TOTAL = sum(_3DMM_DIMS.values())  # 334


def _is_video(fname: str) -> bool:
    return fname.lower().endswith(_VIDEO_EXTENSIONS)


def _is_image(fname: str) -> bool:
    return fname.lower().endswith(_IMG_EXTENSIONS)


def _load_npy_parts_from_dir(npy_dir: str):
    """Load {key}.npy files from a directory. Returns tensor (334,) or None."""
    parts = []
    for key in _3DMM_KEYS:
        path = os.path.join(npy_dir, f"{key}.npy")
        if not os.path.exists(path):
            return None
        arr = np.load(path).astype(np.float32).flatten()
        expected = _3DMM_DIMS[key]
        arr = arr[:expected] if arr.shape[0] >= expected else np.pad(arr, (0, expected - arr.shape[0]))
        parts.append(arr)
    return torch.from_numpy(np.concatenate(parts))  # (334,)


def _load_3dmm_for_frame(stem: str, folder: str):
    """
    Load 3DMM params for a single frame. Two layouts are supported:
      A) Flat:   folder/{stem}_{key}.npy
      B) Nested: folder/frame_dir/{subdir}/{key}.npy
    """
    # Layout A — flat files in the same video_dir
    flat_ok = all(os.path.exists(os.path.join(folder, f"{stem}_{key}.npy")) for key in _3DMM_KEYS)
    if flat_ok:
        parts = []
        for key in _3DMM_KEYS:
            arr = np.load(os.path.join(folder, f"{stem}_{key}.npy")).astype(np.float32).flatten()
            expected = _3DMM_DIMS[key]
            arr = arr[:expected] if arr.shape[0] >= expected else np.pad(arr, (0, expected - arr.shape[0]))
            parts.append(arr)
        return torch.from_numpy(np.concatenate(parts))

    # Layout B — look for a subfolder under frame_dir that holds {key}.npy
    try:
        for subdir in sorted(os.listdir(folder)):
            subdir_path = os.path.join(folder, subdir)
            if os.path.isdir(subdir_path):
                result = _load_npy_parts_from_dir(subdir_path)
                if result is not None:
                    return result
    except OSError:
        pass

    return None


def _read_video_frames(video_path: str, frame_step: int, max_frames: int, start_frame: int = 0):
    if not _HAS_CV2:
        raise ImportError(
            "opencv-python is required to read video files. "
            "Install it with: pip install opencv-python"
        )

    cap = _cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    if start_frame > 0:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, start_frame)

    pil_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(frame_rgb))

            if max_frames is not None and len(pil_frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return pil_frames


class VideoDataset(Dataset):
    """
    Dataset that returns a video as a sequence of frames.

    Args:
        root:          Dataset root path
        split:         Split subfolder name, e.g. 'train', 'test', 'validation'
        transform:     torchvision transform applied per frame (PIL -> Tensor)
        use_3dmm:      Whether to load 3DMM params (frame-folder mode only)
        max_frames:    Max number of frames (None = unlimited)
        frame_step:    Keep 1 frame every N frames (temporal downsampling)
        class_names:   Explicit class order (None = inferred from folders)
        stats_path:    .npz file with mean/std used to normalize 3DMM
        clip_flip_p:   Horizontal flip probability, applied consistently to the
                       WHOLE clip (0.0 = off; enable for train only)
        random_temporal_crop: When a video is longer than max_frames, pick a
                       random start instead of always starting at 0 (train only)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        use_3dmm: bool = False,
        max_frames: int = None,
        frame_step: int = 1,
        class_names: list = None,
        stats_path: str = None,
        clip_flip_p: float = 0.0,
        random_temporal_crop: bool = False,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.use_3dmm = use_3dmm
        self.max_frames = max_frames
        self.frame_step = max(1, frame_step)
        self.clip_flip_p = clip_flip_p
        self.random_temporal_crop = random_temporal_crop

        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Infer class names from folders
        if class_names is None:
            class_names = sorted([
                d for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ])
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        # 3DMM normalization stats
        self._3dmm_mean = None
        self._3dmm_std = None
        if use_3dmm and stats_path and os.path.exists(stats_path):
            stats = np.load(stats_path)
            self._3dmm_mean = torch.from_numpy(stats["mean"].astype(np.float32))
            self._3dmm_std  = torch.from_numpy(stats["std"].astype(np.float32))

        # Build sample list with auto-detected mode
        self.samples = []
        self.mode = None   # 'video_file' or 'frame_folder'
        self._scan(split_dir)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan(self, split_dir: str):
        detected_mode = None

        for cls_name in self.class_names:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label = self.class_to_idx[cls_name]

            for entry in sorted(os.listdir(cls_dir)):
                entry_path = os.path.join(cls_dir, entry)

                if os.path.isfile(entry_path) and _is_video(entry):
                    # video-file mode
                    if detected_mode is None:
                        detected_mode = "video_file"
                    self.samples.append({
                        "mode":  "video_file",
                        "path":  entry_path,
                        "label": label,
                    })

                elif os.path.isdir(entry_path):
                    # frame-folder mode
                    frames = self._list_frames(entry_path)
                    if not frames:
                        continue
                    if detected_mode is None:
                        detected_mode = "frame_folder"
                    self.samples.append({
                        "mode":        "frame_folder",
                        "path":        entry_path,
                        "label":       label,
                        "frame_paths": frames,
                    })

        self.mode = detected_mode or "video_file"

    def _list_frames(self, video_dir: str):
        """
        Return the sorted frame paths (with frame_step applied).

        Two layouts are supported:
          A) Flat:   video_dir/frame_001.jpg, frame_002.jpg, ...
          B) Nested: video_dir/0001/cropped_image.png, 0002/cropped_image.png, ...
             (each frame lives in its own subfolder; image name is arbitrary)
        """
        # Layout A — images directly inside video_dir.
        # max_frames is applied in __getitem__ to support random temporal crop.
        direct_imgs = sorted(f for f in os.listdir(video_dir) if _is_image(f))
        if direct_imgs:
            direct_imgs = direct_imgs[:: self.frame_step]
            return [os.path.join(video_dir, f) for f in direct_imgs]

        # Layout B — each frame is a subfolder containing one image
        subdirs = sorted(
            d for d in os.listdir(video_dir)
            if os.path.isdir(os.path.join(video_dir, d))
        )
        nested_paths = []
        for sub in subdirs:
            sub_path = os.path.join(video_dir, sub)
            imgs = sorted(f for f in os.listdir(sub_path) if _is_image(f))
            if imgs:
                nested_paths.append(os.path.join(sub_path, imgs[0]))

        nested_paths = nested_paths[:: self.frame_step]
        return nested_paths

    def _temporal_crop(self, n_frames: int):
        """Return (start, end) indices bounded by max_frames; random start on train."""
        if self.max_frames is None or n_frames <= self.max_frames:
            return 0, n_frames
        if self.random_temporal_crop:
            start = int(torch.randint(0, n_frames - self.max_frames + 1, (1,)).item())
        else:
            start = 0
        return start, start + self.max_frames

    def _clip_flip(self):
        return self.clip_flip_p > 0 and torch.rand(1).item() < self.clip_flip_p

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        if item["mode"] == "video_file":
            return self._load_video_file(item)
        else:
            return self._load_frame_folder(item)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_video_file(self, item):
        # Random temporal crop: pick a random start frame (cv2 can seek)
        start_frame = 0
        if self.random_temporal_crop and self.max_frames is not None and _HAS_CV2:
            cap = _cv2.VideoCapture(item["path"])
            total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
            needed = self.max_frames * self.frame_step
            if total > needed:
                start_frame = int(torch.randint(0, total - needed + 1, (1,)).item())

        pil_frames = _read_video_frames(
            item["path"], self.frame_step, self.max_frames, start_frame
        )

        if not pil_frames:
            # Empty video — return one black frame so collate doesn't crash
            dummy = Image.new("RGB", (224, 224))
            pil_frames = [dummy]

        flip = self._clip_flip()
        frames = []
        for img in pil_frames:
            if flip:
                img = TF.hflip(img)
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames, dim=0)  # (T, 3, H, W)

        result = {
            "frames": frames,
            "label":  torch.tensor(item["label"], dtype=torch.long),
            "length": torch.tensor(len(pil_frames), dtype=torch.long),
        }

        # 3DMM is not available in video-file mode (no .npy files)
        if self.use_3dmm:
            result["frames_3d"] = torch.zeros(len(pil_frames), _3DMM_TOTAL)

        return result

    def _load_frame_folder(self, item):
        frame_paths = item["frame_paths"]
        video_dir   = item["path"]

        start, end = self._temporal_crop(len(frame_paths))
        frame_paths = frame_paths[start:end]

        flip = self._clip_flip()
        frames    = []
        frames_3d = [] if self.use_3dmm else None

        for fpath in frame_paths:
            img = Image.open(fpath).convert("RGB")
            if flip:
                img = TF.hflip(img)
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)

            if self.use_3dmm:
                stem = os.path.splitext(os.path.basename(fpath))[0]
                # Prefer the frame's own directory (nested layout B);
                # fall back to video_dir (flat layout A)
                frame_dir = os.path.dirname(fpath)
                x_3d = _load_3dmm_for_frame(stem, frame_dir)
                if x_3d is None and frame_dir != video_dir:
                    x_3d = _load_3dmm_for_frame(stem, video_dir)
                if x_3d is None:
                    x_3d = torch.zeros(_3DMM_TOTAL, dtype=torch.float32)
                elif self._3dmm_mean is not None:
                    x_3d = (x_3d - self._3dmm_mean) / (self._3dmm_std + 1e-8)
                frames_3d.append(x_3d)

        frames = torch.stack(frames, dim=0)  # (T, 3, H, W)

        result = {
            "frames": frames,
            "label":  torch.tensor(item["label"], dtype=torch.long),
            "length": torch.tensor(len(frame_paths), dtype=torch.long),
        }

        if self.use_3dmm:
            result["frames_3d"] = torch.stack(frames_3d, dim=0)  # (T, 334)

        return result


# ------------------------------------------------------------------
# Collate
# ------------------------------------------------------------------

def collate_video_fn(batch):
    """
    Custom collate_fn for VideoDataset — zero-pad variable-length sequences.

    Returns dict:
        video:    (B, T_max, 3, H, W)
        labels:   (B,)
        lengths:  (B,)
        video_3d: (B, T_max, 334) if present, absent otherwise
    """
    max_len = max(item["length"].item() for item in batch)

    videos, labels, lengths = [], [], []
    videos_3d = [] if "frames_3d" in batch[0] else None

    for item in batch:
        frames = item["frames"]       # (T, C, H, W)
        T = frames.shape[0]

        if T < max_len:
            pad = torch.zeros(max_len - T, *frames.shape[1:], dtype=frames.dtype)
            frames = torch.cat([frames, pad], dim=0)

        videos.append(frames)
        labels.append(item["label"])
        lengths.append(item["length"])

        if videos_3d is not None:
            f3d = item["frames_3d"]   # (T, 334)
            if T < max_len:
                pad3d = torch.zeros(max_len - T, f3d.shape[1], dtype=f3d.dtype)
                f3d = torch.cat([f3d, pad3d], dim=0)
            videos_3d.append(f3d)

    out = {
        "video":   torch.stack(videos),    # (B, T_max, 3, H, W)
        "labels":  torch.stack(labels),    # (B,)
        "lengths": torch.stack(lengths),   # (B,)
    }
    if videos_3d is not None:
        out["video_3d"] = torch.stack(videos_3d)   # (B, T_max, 334)

    return out


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------

def compute_video_3dmm_stats(root: str, split: str = "train",
                             output: str = "video_3dmm_stats.npz"):
    """
    Compute mean/std of the 3DMM params (334,) over all frames of a split and
    save them to a .npz file with keys 'mean' and 'std' — the exact format
    VideoDataset reads via stats_path.

    Uses the same _load_3dmm_for_frame as training, so both layouts are
    supported (flat {stem}_{key}.npy and nested subdir/{key}.npy). Frames
    missing 3DMM are skipped (not counted as zero-vectors) and reported.
    """
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise RuntimeError(f"Missing directory: {split_dir}")

    from tqdm import tqdm

    # Running sum / sum-of-squares to avoid holding all frames in RAM
    total = np.zeros(_3DMM_TOTAL, dtype=np.float64)
    total_sq = np.zeros(_3DMM_TOTAL, dtype=np.float64)
    n_frames, n_missing, n_videos = 0, 0, 0

    classes = sorted(d for d in os.listdir(split_dir)
                     if os.path.isdir(os.path.join(split_dir, d)))

    # Collect video dirs up front so tqdm has an accurate total/ETA
    video_dirs = []
    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        for entry in sorted(os.listdir(cls_dir)):
            video_dir = os.path.join(cls_dir, entry)
            if os.path.isdir(video_dir):
                video_dirs.append(video_dir)

    pbar = tqdm(video_dirs, desc=f"3DMM stats [{split}]", unit="video",
                dynamic_ncols=True)

    for video_dir in pbar:
        # List frames like _list_frames (flat layout A / nested layout B)
        frame_paths = [os.path.join(video_dir, f)
                       for f in sorted(os.listdir(video_dir)) if _is_image(f)]
        if not frame_paths:
            for sub in sorted(os.listdir(video_dir)):
                sub_path = os.path.join(video_dir, sub)
                if not os.path.isdir(sub_path):
                    continue
                imgs = sorted(f for f in os.listdir(sub_path) if _is_image(f))
                if imgs:
                    frame_paths.append(os.path.join(sub_path, imgs[0]))
        if not frame_paths:
            continue
        n_videos += 1

        for fpath in frame_paths:
            stem = os.path.splitext(os.path.basename(fpath))[0]
            frame_dir = os.path.dirname(fpath)
            x_3d = _load_3dmm_for_frame(stem, frame_dir)
            if x_3d is None and frame_dir != video_dir:
                x_3d = _load_3dmm_for_frame(stem, video_dir)
            if x_3d is None:
                n_missing += 1
                continue

            arr = x_3d.numpy().astype(np.float64)
            total += arr
            total_sq += arr * arr
            n_frames += 1

        pbar.set_postfix({"frames": n_frames, "missing": n_missing})

    pbar.close()

    if n_frames == 0:
        raise RuntimeError(f"No 3DMM params found in {split_dir}")

    mean = total / n_frames
    var = total_sq / n_frames - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    std[std < 1e-6] = 1.0

    out_path = os.path.join(root, output)
    np.savez(out_path,
             mean=mean.astype(np.float32),
             std=std.astype(np.float32))

    print(f"Videos: {n_videos}  Frames: {n_frames}  Missing 3DMM: {n_missing}")
    print(f"mean: {mean.shape}  std: {std.shape}")
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Compute 3DMM mean/std for VideoDataset")
    parser.add_argument("--root", type=str, required=True,
                        help="Dataset root directory (contains the train/ subfolder)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default="video_3dmm_stats.npz")
    args = parser.parse_args()

    compute_video_3dmm_stats(args.root, args.split, args.output)
