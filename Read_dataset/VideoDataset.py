"""
VideoDataset — Dataset cho video facial expression recognition.

Tự động nhận diện 2 chế độ dựa trên cấu trúc thư mục:

  Chế độ A — video-file (ví dụ CAER):
    root/split/ClassName/0001.avi
    root/split/ClassName/0002.avi

  Chế độ B — frame-folder (frame đã extract sẵn):
    root/split/ClassName/video_001/frame_001.jpg
    root/split/ClassName/video_001/frame_002.jpg
    root/split/ClassName/video_001/frame_001_shape.npy  # 3DMM optional

Khi use_3dmm=False, các file .npy không cần thiết.
"""

import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import torchvision.io as _tvio
    # Kiểm tra thực sự có PyAV chưa (read_video cần PyAV)
    _tvio._check_av_available if hasattr(_tvio, "_check_av_available") else None
    _HAS_TVIO = True
except Exception:
    _HAS_TVIO = False

_VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv", ".webm")
_IMG_EXTENSIONS   = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Thứ tự và kích thước các thành phần 3DMM (tổng 334)
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
    Tải 3DMM cho một frame. Hỗ trợ 2 cấu trúc:
      A) Flat: folder/{stem}_{key}.npy  (cấu trúc cũ)
      B) Nested: folder/frame_dir/{subdir}/{key}.npy  (cấu trúc CAER 3DMM)
         — khi frame_path là folder/frame_dir/image.png, gọi với folder=frame_dir
    """
    # Thử cấu trúc A — flat trong cùng video_dir
    flat_ok = all(os.path.exists(os.path.join(folder, f"{stem}_{key}.npy")) for key in _3DMM_KEYS)
    if flat_ok:
        parts = []
        for key in _3DMM_KEYS:
            arr = np.load(os.path.join(folder, f"{stem}_{key}.npy")).astype(np.float32).flatten()
            expected = _3DMM_DIMS[key]
            arr = arr[:expected] if arr.shape[0] >= expected else np.pad(arr, (0, expected - arr.shape[0]))
            parts.append(arr)
        return torch.from_numpy(np.concatenate(parts))

    # Thử cấu trúc B — tìm subfolder trong frame_dir chứa {key}.npy
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
    """
    Đọc frames từ file video. Ưu tiên cv2, fallback sang torchvision.io (cần PyAV).
    Trả về list of PIL.Image.
    """
    if _HAS_CV2:
        return _read_video_cv2(video_path, frame_step, max_frames, start_frame)
    if _HAS_TVIO:
        return _read_video_tvio(video_path, frame_step, max_frames, start_frame)
    raise ImportError(
        "Cần opencv-python hoặc torchvision + PyAV để đọc file video. "
        "Cài đặt: pip install opencv-python"
    )


def _read_video_cv2(video_path: str, frame_step: int, max_frames: int, start_frame: int = 0):
    """Đọc video bằng cv2.VideoCapture."""
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
            # cv2 đọc BGR → chuyển sang RGB
            frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(frame_rgb))

            if max_frames is not None and len(pil_frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return pil_frames


def _read_video_tvio(video_path: str, frame_step: int, max_frames: int, start_frame: int = 0):
    """Đọc video bằng torchvision.io (cần PyAV)."""
    frames_tensor, _, _ = _tvio.read_video(video_path, pts_unit="sec")
    if frames_tensor.shape[0] == 0:
        return []

    frames_tensor = frames_tensor[start_frame::frame_step]
    if max_frames is not None:
        frames_tensor = frames_tensor[:max_frames]

    return [Image.fromarray(frames_tensor[i].numpy()) for i in range(frames_tensor.shape[0])]


class VideoDataset(Dataset):
    """
    Dataset trả về video dưới dạng sequence frames.

    Args:
        root:          Đường dẫn gốc dataset
        split:         Tên subfolder split, vd 'train', 'test', 'validation'
        transform:     torchvision transform áp dụng lên từng frame (PIL → Tensor)
        use_3dmm:      Có tải 3DMM params không (chỉ hỗ trợ ở frame-folder mode)
        max_frames:    Giới hạn số frame tối đa (None = không giới hạn)
        frame_step:    Lấy 1 frame mỗi N frames (temporal downsampling)
        class_names:   Danh sách tên lớp theo thứ tự cụ thể (None = tự suy ra từ folder)
        stats_path:    File .npz chứa mean/std để normalize 3DMM
        clip_flip_p:   Xác suất horizontal flip — áp dụng nhất quán cho TOÀN BỘ clip
                       (0.0 = tắt; chỉ bật cho train)
        random_temporal_crop: Khi video dài hơn max_frames, chọn điểm bắt đầu ngẫu nhiên
                       thay vì luôn lấy từ đầu (chỉ bật cho train)
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

        # Tự suy ra class names
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

        # Xây danh sách samples — tự phát hiện chế độ
        self.samples = []  # list of dict
        self.mode = None   # 'video_file' hoặc 'frame_folder'
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
                    # ---- chế độ video-file ----
                    if detected_mode is None:
                        detected_mode = "video_file"
                    self.samples.append({
                        "mode":       "video_file",
                        "path":       entry_path,
                        "label":      label,
                    })

                elif os.path.isdir(entry_path):
                    # ---- chế độ frame-folder ----
                    frames = self._list_frames(entry_path)
                    if not frames:
                        continue
                    if detected_mode is None:
                        detected_mode = "frame_folder"
                    self.samples.append({
                        "mode":       "frame_folder",
                        "path":       entry_path,
                        "label":      label,
                        "frame_paths": frames,
                    })

        self.mode = detected_mode or "video_file"

    def _list_frames(self, video_dir: str):
        """
        Trả về danh sách đường dẫn frame đã được sắp xếp (với frame_step).

        Hỗ trợ 2 cấu trúc:
          A) Flat: video_dir/frame_001.jpg, frame_002.jpg, ...
          B) Nested: video_dir/0001/cropped_image.png, 0002/cropped_image.png, ...
             (mỗi frame nằm trong subfolder riêng, ảnh có thể đặt tên bất kỳ)
        """
        # Thử cấu trúc A — ảnh nằm trực tiếp trong video_dir
        # Lưu ý: max_frames được áp dụng lúc __getitem__ (hỗ trợ random temporal crop)
        direct_imgs = sorted(f for f in os.listdir(video_dir) if _is_image(f))
        if direct_imgs:
            direct_imgs = direct_imgs[:: self.frame_step]
            return [os.path.join(video_dir, f) for f in direct_imgs]

        # Thử cấu trúc B — mỗi frame là một subfolder chứa một ảnh
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
        """Trả về (start, end) index theo max_frames; random start khi train."""
        if self.max_frames is None or n_frames <= self.max_frames:
            return 0, n_frames
        if self.random_temporal_crop:
            start = int(torch.randint(0, n_frames - self.max_frames + 1, (1,)).item())
        else:
            start = 0
        return start, start + self.max_frames

    def _clip_flip(self):
        """Quyết định flip một lần cho cả clip (nhất quán giữa các frame)."""
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
        # Random temporal crop: chọn frame bắt đầu ngẫu nhiên (cv2 seek được,
        # tvio đọc cả video rồi slice)
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
            # Video rỗng — trả về 1 frame đen để không crash collate
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
            "frames":  frames,
            "label":   torch.tensor(item["label"], dtype=torch.long),
            "length":  torch.tensor(len(pil_frames), dtype=torch.long),
        }

        # 3DMM không hỗ trợ ở chế độ video-file (chưa có file .npy)
        if self.use_3dmm:
            result["frames_3d"] = torch.zeros(len(pil_frames), _3DMM_TOTAL)

        return result

    def _load_frame_folder(self, item):
        frame_paths = item["frame_paths"]
        video_dir   = item["path"]

        start, end = self._temporal_crop(len(frame_paths))
        frame_paths = frame_paths[start:end]

        flip = self._clip_flip()
        frames   = []
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
                # Prefer frame's own directory (nested structure B);
                # fall back to video_dir (flat structure A)
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
            "frames":  frames,
            "label":   torch.tensor(item["label"], dtype=torch.long),
            "length":  torch.tensor(len(frame_paths), dtype=torch.long),
        }

        if self.use_3dmm:
            result["frames_3d"] = torch.stack(frames_3d, dim=0)  # (T, 334)

        return result


# ------------------------------------------------------------------
# Collate
# ------------------------------------------------------------------

def collate_video_fn(batch):
    """
    Custom collate_fn cho VideoDataset — zero-pad variable-length sequences.

    Returns dict:
        video:    (B, T_max, 3, H, W)
        labels:   (B,)
        lengths:  (B,)
        video_3d: (B, T_max, 334) nếu có, absent nếu không
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
