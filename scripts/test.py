
# scripts/test.py
import os, argparse, json, csv
import torch
from torch.utils.data import DataLoader
from dataset import NoseDataset
from utils import load_checkpoint, euclidean_errors, summarize_errors
from models.snoutnet import SnoutNet
from models.snoutnet_alexnet import SnoutNetAlexNet
from models.snoutnet_vgg import SnoutNetVGG16

def build_model(model_name: str, pretrained: bool):
    if model_name == "snoutnet":
        return SnoutNet()
    elif model_name == "alexnet":
        return SnoutNetAlexNet(pretrained=pretrained)
    elif model_name == "vgg16":
        return SnoutNetVGG16(pretrained=pretrained)
    else:
        raise ValueError(model_name)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (*.pt)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_csv", type=str, default="eval.csv")
    args = p.parse_args()

    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    model_name = ckpt["model"]
    pretrained = ckpt.get("pretrained", False)
    mean, std = ckpt.get("norm", ((0.5,)*3, (0.5,)*3))
    model = build_model(model_name, pretrained=pretrained)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dataset = NoseDataset(args.data_root, "test-noses.txt", input_size=227, augment=None, normalize=(mean, std))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_errors = []
    rows = [("file","gt_u","gt_v","pred_u","pred_v","error")]
    with torch.no_grad():
        for img, target, meta in loader:
            pred = model(img)
            err = euclidean_errors(pred, target)
            for i in range(img.size(0)):
                gt_u, gt_v = target[i].tolist()
                pu, pv = pred[i].tolist()
                rows.append((meta["file"][i], gt_u, gt_v, pu, pv, float(err[i])))
            all_errors.append(err)
    all_errors = torch.cat(all_errors, dim=0)
    stats = summarize_errors(all_errors)
    print("Evaluation (pixels):", json.dumps(stats, indent=2))

    with open(args.save_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("Wrote", args.save_csv)

if __name__ == "__main__":
    main()
