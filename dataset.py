
# dataset.py
import os, math, random, re
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

LABEL_LINE_RE = re.compile(r'^\s*([^,]+)\s*,\s*"\((\d+)\s*,\s*(\d+)\)"')

def parse_label_line(line: str) -> Tuple[str, Tuple[int, int]]:
    m = LABEL_LINE_RE.match(line)
    if not m:
        raise ValueError(f"Bad label line: {line}")
    fname = m.group(1)
    u = int(m.group(2))
    v = int(m.group(3))
    return fname, (u, v)

class NoseDataset(Dataset):
    """
    Custom dataset for ELEC 475 Lab 2.
    Expects directory with:
      images-original/images/<files>.jpg
      train-noses.txt / test-noses.txt listing file name and "(u, v)"
    Transforms/augmentations (optional) are applied consistently to image and uv.
    """
    def __init__(
        self,
        root: str,
        split_file: str,
        input_size: int = 227,
        augment: Optional[Dict[str, Any]] = None,
        normalize: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((0.5,)*3, (0.5,)*3),
    ):
        self.root = root
        self.img_dir = os.path.join(root, "images-original", "images")
        self.split_file = os.path.join(root, split_file)
        self.input_size = input_size
        self.augment_cfg = augment or {}
        self.normalize = normalize
        self.samples = []
        with open(self.split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                fname, (u, v) = parse_label_line(line)
                self.samples.append((fname, (u, v)))
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples parsed from {self.split_file}")

    def __len__(self):
        return len(self.samples)

    def _maybe_hflip(self, img, uv):
        if self.augment_cfg.get("hflip", False) and random.random() < 0.5:
            w, h = img.size
            img = TF.hflip(img)
            u, v = uv
            uv = (w - 1 - u, v)
        return img, uv

    def _maybe_rotate(self, img, uv):
        degrees = self.augment_cfg.get("rotation_deg", 0.0)
        if not degrees:
            return img, uv
        angle = random.uniform(-degrees, degrees)
        w, h = img.size
        img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        u, v = uv
        x, y = (u - cx), (v - cy)
        rad = -angle * math.pi / 180.0
        xr = x * math.cos(rad) - y * math.sin(rad)
        yr = x * math.sin(rad) + y * math.cos(rad)
        ur, vr = xr + cx, yr + cy
        return img, (ur, vr)

    def _maybe_color_jitter(self, img):
        b = self.augment_cfg.get("brightness", 0.0)
        c = self.augment_cfg.get("contrast", 0.0)
        if b:
            factor = 1.0 + random.uniform(-b, b)
            img = TF.adjust_brightness(img, factor)
        if c:
            factor = 1.0 + random.uniform(-c, c)
            img = TF.adjust_contrast(img, factor)
        return img

    def __getitem__(self, idx):
        fname, (u, v) = self.samples[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size

        img, (u, v) = self._maybe_hflip(img, (u, v))
        img, (u, v) = self._maybe_rotate(img, (u, v))
        img = self._maybe_color_jitter(img)

        img = TF.resize(img, [self.input_size, self.input_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        w1, h1 = self.input_size, self.input_size
        scale_u = w1 / w0
        scale_v = h1 / h0
        u = u * scale_u
        v = v * scale_v

        tensor = TF.to_tensor(img)
        mean, std = self.normalize
        tensor = TF.normalize(tensor, mean, std)

        target = torch.tensor([u, v], dtype=torch.float32)
        meta = {"file": fname, "orig_size": (w0, h0)}
        return tensor, target, meta
