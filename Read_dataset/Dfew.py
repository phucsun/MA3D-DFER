"""
DfewDataset — dataset for video facial expression recognition (DFEW Format).

Cấu trúc mới:
    - Root: "Datasets"
    - Ảnh: Datasets/DFEW/{video_id}/{video_id}_{frame_id}.jpg
    - 3DMM: Datasets/dfew_3dmm/{video_id}/{video_id}_{frame_id}/{video_id}_{frame_id}/*.npy
    - Nhãn: Datasets/dfew/extracted/DFEW-part2/EmoLabel_DataSplit/{split}(single-labeled)/set_1.csv
"""

import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_3DMM_KEYS = ["shape", "tex", "exp", "pose", "detail"]
_3DMM_DIMS = {"shape": 100, "tex": 50, "exp": 50, "pose": 6, "detail": 128}
_3DMM_TOTAL = sum(_3DMM_DIMS.values())  # 334


def _is_image(fname: str) -> bool:
    return fname.lower().endswith(_IMG_EXTENSIONS)


def _load_3dmm_for_frame(root: str, vid_str: str, frame_stem: str):
    """
    Load 3DMM params cho cấu trúc cụ thể:
    Datasets/dfew_3dmm/{vid_str}/{frame_stem}/{frame_stem}/{key}.npy
    Ví dụ: Datasets/dfew_3dmm/00001/00001_00001/00001_00001/shape.npy
    """
    npy_dir = os.path.join(root, "dfew_3dmm", vid_str, frame_stem, frame_stem)
    
    if not os.path.isdir(npy_dir):
        return None

    parts = []
    for key in _3DMM_KEYS:
        path = os.path.join(npy_dir, f"{key}.npy")
        if not os.path.exists(path):
            return None
            
        arr = np.load(path).astype(np.float32).flatten()
        expected = _3DMM_DIMS[key]
        arr = arr[:expected] if arr.shape[0] >= expected else np.pad(arr, (0, expected - arr.shape[0]))
        parts.append(arr)
        
    return torch.from_numpy(np.concatenate(parts))


class DfewDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        use_3dmm: bool = False,
        max_frames: int = None,
        frame_step: int = 1,
        sample_strategy: str = "normal", # "normal", "uniform", "segment", "OF"
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
        self.sample_strategy = sample_strategy
        self.clip_flip_p = clip_flip_p
        self.random_temporal_crop = random_temporal_crop
        self.class_names = ["Happy", "Sad", "Neutral", "Angry", "Surprise", "Disgust", "Fear"]

        # Load 3DMM normalization stats
        self._3dmm_mean = None
        self._3dmm_std = None
        if use_3dmm and stats_path and os.path.exists(stats_path):
            print(f"Loading 3DMM stats from: {stats_path}")
            stats = np.load(stats_path)
            self._3dmm_mean = torch.from_numpy(stats["mean"].astype(np.float32))
            self._3dmm_std  = torch.from_numpy(stats["std"].astype(np.float32))

        self.samples = []
        self._scan()

    def _scan(self):
        # Đường dẫn tới file CSV tương ứng với split
        csv_dir = os.path.join(self.root, "dfew", "extracted", "DFEW-part2", "EmoLabel_DataSplit", f"{self.split}(single-labeled)")
        csv_path = os.path.join(csv_dir, "set_1.csv")
        
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file nhãn: {csv_path}")

        # Thư mục chứa ảnh gốc
        dfew_imgs_dir = os.path.join(self.root, "DFEW")

        # Đọc file CSV
        with open(csv_path, 'r') as f:
            lines = f.read().strip().splitlines()
        
        # Bỏ qua header
        for line in lines[1:]:
            # Tách bằng khoảng trắng hoặc dấu phẩy
            parts = re.split(r'[\s,]+', line.strip())
            if len(parts) < 2:
                continue
                
            vid_id = int(parts[0])
            label = int(parts[1]) - 1 # LƯU Ý: Trừ 1 nếu nhãn của DFEW bắt đầu từ 1 (để PyTorch nhận 0-indexed)

            vid_str = f"{vid_id:05d}"  # Chuyển 1 -> "00001"
            vid_dir = os.path.join(dfew_imgs_dir, vid_str)

            if not os.path.isdir(vid_dir):
                continue

            # Lấy toàn bộ frames trong thư mục video (e.g. 00001_00001.jpg)
            frames = sorted([f for f in os.listdir(vid_dir) if _is_image(f)])
            if not frames:
                continue

            # Downsample khung hình theo frame_step
            frames = frames[::self.frame_step]
            frame_paths = [os.path.join(vid_dir, f) for f in frames]

            self.samples.append({
                "vid_str": vid_str,
                "label": label,
                "frame_paths": frame_paths,
            })

    def _sample_indices(self, num_frames: int, frame_path: str = None):
        """Trả về list các indices của frame sẽ được lấy dựa trên strategy"""
        
        # 1. NORMAL: Cắt 1 đoạn liên tục (Giữ nguyên logic cũ của bạn)
        if self.sample_strategy == "normal":
            if self.max_frames is None or num_frames <= self.max_frames:
                return list(range(num_frames))
            
            if self.random_temporal_crop:
                start = int(torch.randint(0, num_frames - self.max_frames + 1, (1,)).item())
            else:
                start = 0
            return list(range(start, start + self.max_frames))

        # Nếu không phải normal mà video ngắn hơn max_frames, 
        # cả uniform và segment đều có thể tự nội suy (duplicate frames) để đủ max_frames.
        if self.max_frames is None:
            return list(range(num_frames))

        # 2. UNIFORM SAMPLING: Lấy max_frames cách đều nhau
        if self.sample_strategy == "uniform":
            # np.linspace tự động chia đều, ép kiểu về int để lấy index
            indices = np.linspace(0, num_frames - 1, self.max_frames, dtype=int)
            # print(f"Uniform sampling: num_frames={num_frames}, max_frames={self.max_frames}, indices={indices}")
            return indices.tolist()

        # 3. TEMPORAL SEGMENT SAMPLING: Chia U đoạn, mỗi đoạn lấy ngẫu nhiên V frames
        elif self.sample_strategy == "segment":
            U = 8
            V = self.max_frames // U

            boundaries = torch.linspace(0, num_frames, U + 1).long()
            indices = []

            # TRAIN
            if self.split == "train":
                for i in range(U):
                    start_idx = boundaries[i].item()
                    end_idx = boundaries[i + 1].item()

                    if start_idx == end_idx:
                        seg_indices = [min(start_idx, num_frames - 1)] * V
                    else:
                        seg_length = end_idx - start_idx
                        candidates = torch.arange(start_idx, end_idx)

                        if seg_length <= V:
                            rand_idx = torch.randint(0, seg_length, (V,))
                        else:
                            rand_idx = torch.randperm(seg_length)[:V]

                        seg_indices = candidates[rand_idx]
                        seg_indices = torch.sort(seg_indices)[0]
                        seg_indices = seg_indices.tolist()

                    indices.extend(seg_indices)
            # TEST
            else:
                for i in range(U):
                    start_idx = boundaries[i].item()
                    end_idx = boundaries[i + 1].item()

                    seg_length = max(end_idx - start_idx, 1)

                    # chia đều V vị trí quanh center segment
                    centers = torch.linspace(
                        start_idx,
                        end_idx - 1,
                        V + 2
                    )[1:-1]  # bỏ biên, lấy V điểm giữa

                    seg_indices = centers.round().long().tolist()

                    # clamp để chắc chắn hợp lệ
                    seg_indices = [
                        min(max(idx, start_idx), num_frames - 1)
                        for idx in seg_indices
                    ]

                    indices.extend(seg_indices)

            return indices  
        elif self.sample_strategy == "OF":
            import json

            json_path = os.path.join("Datasets", "dfew_on_apex_off.json")

            with open(json_path, "r") as f:
                of_data = json.load(f)

            # Datasets/DFEW/images/00001/000001_000001.jpg
            video_id = os.path.basename(os.path.dirname(frame_path))
            if video_id not in of_data:
                # fallback
                print(f"[Warning] Video {video_id} không có thông tin OF, dùng uniform sampling.")
                return list(np.linspace(0, num_frames - 1, self.num_frames, dtype=int))

            info = of_data[video_id]

            onset = info["onset"]
            apex_list = info["apex"]
            offset = info["offset"]

            indices = []

            indices.append(onset)
            indices.append(offset)
            n_apex = len(apex_list)

            # Thêm toàn bộ apex trước
            for apex in apex_list:
                indices.append(apex)

            # Số frame còn lại dành cho các frame lân cận
            remain = self.max_frames - len(indices)

            if remain <= 0:
                return sorted(set(np.clip(indices, 0, self.max_frames - 1)))

            base = remain // n_apex
            extra = remain % n_apex

            for i, apex in enumerate(apex_list):

                # k = số frame lân cận của apex này
                k = base + (1 if i < extra else 0)

                if k <= 0:
                    continue

                left = k // 2
                right = k - left

                # khoảng cách tối đa tới onset / offset
                left_dist = apex - onset
                right_dist = offset - apex

                # step động
                left_step = 4 if left == 0 else max(1, min(4, left_dist // (left + 1)))
                right_step = 4 if right == 0 else max(1, min(4, right_dist // (right + 1)))

                # bên trái
                for r in range(left, 0, -1):
                    idx = apex - r * left_step
                    indices.append(int(np.clip(idx, onset, offset)))

                # bên phải
                for r in range(1, right + 1):
                    idx = apex + r * right_step
                    indices.append(int(np.clip(idx, onset, offset)))

            indices = sorted(set(indices))

            # If duplicates cause too few frames, pad with neighbors
            while len(indices) < self.max_frames:
                added = False
                for idx in list(indices):
                    for d in (-1, 1):
                        x = idx + d
                        if 0 <= x < num_frames and x not in indices:
                            indices.append(x)
                            added = True
                            if len(indices) == self.max_frames:
                                break
                    if len(indices) == self.max_frames:
                        break
                if not added:
                    break

            indices = sorted(indices)
            # print(f"[OF Sampling] video={video_id}, self.max_frames={self.max_frames}, indices={indices}")
            # print(f"[OF Sampling] onset={onset}, apex={apex_list}, offset={offset}, remain={remain}, base={base}, extra={extra}")
            return indices
        else:
            raise ValueError(f"Chiến lược không hợp lệ: {self.sample_strategy}")

    def _clip_flip(self):
        return self.clip_flip_p > 0 and torch.rand(1).item() < self.clip_flip_p

    def __len__(self):
        return len(self.samples)
    
    def get_sampled_frame_paths(self, idx):
        item = self.samples[idx]
        frame_paths = item["frame_paths"]

        indices = self._sample_indices(
            len(frame_paths),
            frame_path=frame_paths[0]
        )

        return {
            "vid_str": item["vid_str"],
            "frame_paths": [frame_paths[i] for i in indices],
        }

    def __getitem__(self, idx):
        item = self.samples[idx]
        frame_paths = item["frame_paths"]
        vid_str = item["vid_str"]

        # --- Áp dụng Sampling Strategy ---
        indices = self._sample_indices(len(frame_paths), frame_path = frame_paths[0])
        sampled_frame_paths = [frame_paths[i] for i in indices]

        flip = self._clip_flip()
        frames    = []
        frames_3d = [] if self.use_3dmm else None

        for fpath in sampled_frame_paths: # Lặp qua danh sách đã được sample
            img = Image.open(fpath).convert("RGB")
            if flip:
                img = TF.hflip(img)
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)

            if self.use_3dmm:
                stem = os.path.splitext(os.path.basename(fpath))[0]
                x_3d = _load_3dmm_for_frame(self.root, vid_str, stem)

                if x_3d is None:
                    print(f"[Missing 3DMM] video={vid_str}, frame={stem}")
                    x_3d = torch.zeros(_3DMM_TOTAL, dtype=torch.float32)

                else:
                    # kiểm tra file npy có toàn số 0 không
                    # if torch.all(x_3d == 0):
                    #     print(f"[All Zero 3DMM] video={vid_str}, frame={stem}")

                    if self._3dmm_mean is not None:
                        x_3d = (x_3d - self._3dmm_mean) / (self._3dmm_std + 1e-8)

                frames_3d.append(x_3d)

        frames = torch.stack(frames, dim=0)

        result = {
            "frames": frames,
            "label":  torch.tensor(item["label"], dtype=torch.long),
            "length": torch.tensor(len(sampled_frame_paths), dtype=torch.long),
        }

        if self.use_3dmm:
            result["frames_3d"] = torch.stack(frames_3d, dim=0)

        return result


# ------------------------------------------------------------------
# Collate Function
# ------------------------------------------------------------------

def collate_video_fn(batch):
    max_len = max(item["length"].item() for item in batch)

    videos, labels, lengths = [], [], []
    videos_3d = [] if "frames_3d" in batch[0] else None

    for item in batch:
        frames = item["frames"]
        T = frames.shape[0]

        if T < max_len:
            pad = torch.zeros(max_len - T, *frames.shape[1:], dtype=frames.dtype)
            frames = torch.cat([frames, pad], dim=0)

        videos.append(frames)
        labels.append(item["label"])
        lengths.append(item["length"])

        if videos_3d is not None:
            f3d = item["frames_3d"]
            if T < max_len:
                pad3d = torch.zeros(max_len - T, f3d.shape[1], dtype=f3d.dtype)
                f3d = torch.cat([f3d, pad3d], dim=0)
            videos_3d.append(f3d)

    out = {
        "video":   torch.stack(videos),    
        "labels":  torch.stack(labels),    
        "lengths": torch.stack(lengths),   
    }
    if videos_3d is not None:
        out["video_3d"] = torch.stack(videos_3d)   

    return out

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def detect_onset_apex_offset(
    frame_paths,
    smooth_sigma=2,
    prominence=0.2,
    distance=25,
    onset_ratio=0.1,
):
    """
    Detect onset, apex and offset using Optical Flow + Peak Detection.

    Parameters
    ----------
    frame_paths : list[str]

    smooth_sigma : float
        Gaussian smoothing sigma.

    prominence : float
        Minimum peak prominence after normalization.

    distance : int
        Minimum distance between neighbouring peaks.

    apex_expand_ratio : float
        Expand apex region until motion falls below
        apex_expand_ratio * peak_value.

    onset_ratio : float
        Motion ratio for onset/offset detection.

    Returns
    -------
    onset : int
    apex : list[int]
    offset : int
    """

    if len(frame_paths) < 2:
        return 0, [0], 0

    # Optical Flow Magnitude
    motion = [0]
    prev = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)

    if prev is None:
        raise ValueError(frame_paths[0])

    for path in frame_paths[1:]:
        curr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if curr is None:
            raise ValueError(path)

        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, pyr_scale=0.5, levels=3,
            winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,)
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        motion.append(mag.mean())
        prev = curr
    motion = np.asarray(motion)

    # Smooth
    motion = gaussian_filter1d(motion, sigma=smooth_sigma)

    # Normalize
    if motion.max() == motion.min():
        return 0, [0], len(frame_paths)-1

    motion = (motion - motion.min()) / (motion.max() - motion.min())


    # Peak Detection
    peaks, properties = find_peaks(
        motion,
        prominence=prominence,
        distance=distance,
    )

    if len(peaks) == 0:
        apex = [int(np.argmax(motion))]
    else:
        apex = peaks.tolist()

    # Onset
    onset_threshold = motion[apex].max() * onset_ratio

    onset = 0

    for i in range(apex[0]):
        if motion[i] >= onset_threshold:
            onset = i
            break

    # Offset
    offset = len(motion)-1

    for i in range(apex[-1], len(motion)):
        if motion[i] <= onset_threshold:
            offset = i
            break

    return onset, apex, offset



import os
import json
from tqdm import tqdm


def build_on_apex_off(root, save_path):
    video_root = os.path.join(root, "DFEW")
    result = {}
    for video_id in tqdm(sorted(os.listdir(video_root))):

        video_dir = os.path.join(video_root, video_id)

        if not os.path.isdir(video_dir):
            continue

        frames = sorted([
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith(".jpg")
        ])

        if len(frames) < 3:
            continue

        onset, apex, offset = detect_onset_apex_offset(frames)

        result[video_id] = {
            "onset": onset,
            "apex": apex,
            "offset": offset
        }

    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)


# ------------------------------------------------------------------
# Stats (Đã được điều chỉnh theo cấu trúc DFEW mới)
# ------------------------------------------------------------------

def compute_video_3dmm_stats_from_dataset(root: str, frame_step: int = 1, batch_size: int = 256, output: str = "video_3dmm_stats.npz"):
    """
    Khởi tạo DfewDataset tạm thời dựa trên tập 'train' để duyệt qua từng video 
    và tính toán stats 3DMM chuẩn xác nhất với cơ chế Batching để tăng tốc.
    """
    # 1. Khởi tạo dataset tạm thời chuyên cho tính toán stats
    stats_dataset = DfewDataset(
        root=root,
        split="train",            # Luôn tính toán trên tập train
        transform=None,           # Không cần transform ảnh vì chỉ tính toán 3DMM
        use_3dmm=False,           # Đặt là False vì ta sẽ chủ động load file gốc, tránh bị normalize ngược
        max_frames=1,          
        frame_step=frame_step,    # Khớp với frame_step dùng khi train
        sample_strategy="OF", 
        stats_path=None,          # Chưa có file stat nên truyền None
        clip_flip_p=0.0,          # Tắt lật ảnh
        random_temporal_crop=False # Không crop thời gian bừa bãi để tính đủ frame
    )

    if len(stats_dataset) == 0:
        raise RuntimeError(f"Dataset trống! Hãy kiểm tra lại đường dẫn root: {root}")

    # Bước cải tiến 1: Thu thập tất cả thông tin các frame cần load trước
    all_frame_tasks = []
    for idx in range(len(stats_dataset)):
        sample = stats_dataset.get_sampled_frame_paths(idx)
        vid_str = sample["vid_str"]
        for fpath in sample["frame_paths"]:
            stem = os.path.splitext(os.path.basename(fpath))[0]
            all_frame_tasks.append((vid_str, stem))

    total_tasks = len(all_frame_tasks)
    if total_tasks == 0:
        raise RuntimeError(f"Không tìm thấy frame nào được quét từ dataset.")

    total = np.zeros(_3DMM_TOTAL, dtype=np.float64)
    total_sq = np.zeros(_3DMM_TOTAL, dtype=np.float64)
    n_frames, n_missing = 0, 0

    # Bước cải tiến 2: Duyệt theo từng Batch thay vì từng frame lẻ
    pbar = tqdm(
        range(0, total_tasks, batch_size), 
        desc=f"Computing 3DMM stats (batch_size={batch_size}, step={frame_step})", 
        unit="batch", 
        dynamic_ncols=True
    )

    for i in pbar:
        batch_tasks = all_frame_tasks[i : i + batch_size]
        batch_arrays = []

        # Load song song/tuần tự các file trong batch hiện tại
        for vid_str, stem in batch_tasks:
            x_3d = _load_3dmm_for_frame(root, vid_str, stem)
            if x_3d is None:
                n_missing += 1
                continue
            
            # Chuyển về numpy array
            arr = x_3d.numpy().astype(np.float64)
            batch_arrays.append(arr)

        if len(batch_arrays) == 0:
            continue

        # Chuyển list thành 1 matrix lớn [Batch_size, 3DMM_Dim] để tính vector hóa
        batch_np = np.stack(batch_arrays, axis=0)
        
        # Cộng dồn một lần duy nhất cho cả batch (Nhanh hơn rất nhiều so với loop từng dòng)
        total += batch_np.sum(axis=0)
        total_sq += (batch_np ** 2).sum(axis=0)
        n_frames += batch_np.shape[0]

        pbar.set_postfix({"frames": n_frames, "missing": n_missing})

    pbar.close()

    if n_frames == 0:
        raise RuntimeError(f"Không tìm thấy params 3DMM nào trong tập Train với frame_step={frame_step}")

    # 3. Tính toán Mean và Std
    mean = total / n_frames
    var = total_sq / n_frames - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    std[std < 1e-6] = 1.0 

    out_path = os.path.join(root, output)
    np.savez(out_path, mean=mean.astype(np.float32), std=std.astype(np.float32))

    print(f"\n[Done] Total Tasks: {total_tasks} | Frames processed: {n_frames} | Missing 3DMM: {n_missing}")
    print(f"Saved stats to: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Compute 3DMM mean/std for VideoDataset")
    parser.add_argument("--root", type=str,  default="Datasets", help="Dataset root directory (vd: 'Datasets')")
    parser.add_argument("--output", type=str, default="video_3dmm_stats_OF.npz")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size để tính toán cho nhanh")
    args = parser.parse_args()

    compute_video_3dmm_stats_from_dataset(root=args.root, frame_step=1, batch_size=args.batch_size, output=args.output)

    # build_on_apex_off(root="Datasets", save_path=os.path.join("Datasets", "dfew_on_apex_off.json"))