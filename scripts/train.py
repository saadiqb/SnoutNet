
# scripts/train.py
import os, argparse, math, time, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import NoseDataset
from utils import set_seed, save_checkpoint, summarize_errors
from models.snoutnet import SnoutNet
from models.snoutnet_alexnet import SnoutNetAlexNet
from models.snoutnet_vgg import SnoutNetVGG16

MODEL_CHOICES = {
    "snoutnet": SnoutNet,
    "alexnet": SnoutNetAlexNet,
    "vgg16": SnoutNetVGG16,
}

def get_norm_from_model(model_name: str):
    if model_name == "snoutnet":
        return ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        tmp = MODEL_CHOICES[model_name](pretrained=True)
        return (tuple(tmp.mean), tuple(tmp.std))

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
    p.add_argument("--data_root", type=str, required=True, help="Path containing images-original/ and train/test txt files")
    p.add_argument("--model", type=str, default="snoutnet", choices=list(MODEL_CHOICES.keys()))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--pretrained", action="store_true", help="Use pretrained weights for AlexNet/VGG16 variants")
    p.add_argument("--aug", action="store_true", help="Enable augmentations")
    p.add_argument("--save", type=str, default="checkpoints")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional: Force training only on GPU
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU - Not Available")


    mean, std = get_norm_from_model(args.model)
    augment = None
    if args.aug:
        augment = {"hflip": True, "rotation_deg": 10.0, "brightness": 0.2, "contrast": 0.2}

    train_set = NoseDataset(args.data_root, "train-noses.txt", input_size=227, augment=augment, normalize=(mean, std))
    test_set  = NoseDataset(args.data_root, "test-noses.txt",  input_size=227, augment=None,    normalize=(mean, std))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    os.makedirs(args.save, exist_ok=True)
    ckpt_path = os.path.join(args.save, f"{args.model}_{'aug' if args.aug else 'plain'}.pt")

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for img, target, _ in train_loader:
            img = img.to(device)
            target = target.to(device)
            pred = model(img)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * img.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for img, target, _ in val_loader:
                img = img.to(device)
                target = target.to(device)
                pred = model(img)
                loss = criterion(pred, target)
                val_running += loss.item() * img.size(0)
        val_loss = val_running / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": args.model,
                        "state_dict": model.state_dict(),
                        "pretrained": args.pretrained,
                        "aug": args.aug,
                        "norm": (mean, std)}, ckpt_path)

    with open(os.path.join(args.save, f"{args.model}_{'aug' if args.aug else 'plain'}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("Saved checkpoint to", ckpt_path)

if __name__ == "__main__":
    main()
