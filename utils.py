
# utils.py
import os, math, random, json
import torch
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, map_location=None):
    return torch.load(path, map_location=map_location)

def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(b, device) for b in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch

def euclidean_errors(pred_uv: torch.Tensor, gt_uv: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred_uv - gt_uv, dim=1)

def summarize_errors(errors: torch.Tensor):
    e = errors.detach().cpu().numpy()
    return {
        "min": float(np.min(e)),
        "max": float(np.max(e)),
        "mean": float(np.mean(e)),
        "stdev": float(np.std(e)),
    }
