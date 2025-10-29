# scripts/ensemble.py
import os, argparse, json, csv
import torch
from torch.utils.data import DataLoader

from dataset import NoseDataset
from utils import load_checkpoint, euclidean_errors, summarize_errors
from models.snoutnet import SnoutNet
from models.snoutnet_alexnet import SnoutNetAlexNet
from models.snoutnet_vgg import SnoutNetVGG16


def build(model_name, pretrained):
    if model_name == "snoutnet":
        m = SnoutNet()
    elif model_name == "alexnet":
        m = SnoutNetAlexNet(pretrained=pretrained)
    elif model_name == "vgg16":
        m = SnoutNetVGG16(pretrained=pretrained)
    else:
        raise ValueError(model_name)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--snoutnet_ckpt", type=str, required=True)
    p.add_argument("--alexnet_ckpt", type=str, required=True)
    p.add_argument("--vgg16_ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_csv", type=str, default=None, help="optional path to write per-image preds/errors")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoints and build models
    ckpts = [
        load_checkpoint(args.snoutnet_ckpt, map_location="cpu"),
        load_checkpoint(args.alexnet_ckpt,  map_location="cpu"),
        load_checkpoint(args.vgg16_ckpt,    map_location="cpu"),
    ]
    # Use norm from the first checkpoint (dataset-level normalization)
    mean, std = ckpts[0].get("norm", ((0.5,)*3, (0.5,)*3))

    models = []
    for ck in ckpts:
        m = build(ck["model"], pretrained=ck.get("pretrained", False))
        m.load_state_dict(ck["state_dict"])
        m.to(device).eval()
        models.append(m)

    dataset = NoseDataset(
        args.data_root,
        "test-noses.txt",
        input_size=227,
        augment=None,
        normalize=(mean, std),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    all_err = []
    rows = [("file", "gt_u", "gt_v", "pred_u", "pred_v", "error")] if args.save_csv else None

    with torch.no_grad():
        for imgs, targets, meta in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Forward all three models
            preds_list = [m(imgs) for m in models]      # list of (B,2)
            fused = torch.stack(preds_list, dim=0).median(dim=0).values  # (B,2)

            err = euclidean_errors(fused, targets)      # (B,)
            all_err.append(err)

            if rows is not None:
                fu, fv = fused[:, 0].detach().cpu(), fused[:, 1].detach().cpu()
                gu, gv = targets[:, 0].detach().cpu(), targets[:, 1].detach().cpu()
                ee = err.detach().cpu()
                files = meta["file"] if isinstance(meta, dict) else meta  # support dict meta
                for i in range(len(files)):
                    rows.append((files[i], float(gu[i]), float(gv[i]),
                                 float(fu[i]), float(fv[i]), float(ee[i])))

    all_err = torch.cat(all_err, dim=0)
    stats = summarize_errors(all_err)
    print("Ensemble evaluation (pixels):", json.dumps(stats, indent=2))

    if rows is not None:
        out = args.save_csv
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print("Wrote", out)


if __name__ == "__main__":
    main()
