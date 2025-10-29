
# scripts/visualize.py
import os, argparse, random
import torch
from PIL import Image, ImageDraw
from dataset import NoseDataset
from utils import load_checkpoint
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

def draw_point(im: Image.Image, u: float, v: float, color, r=5):
    d = ImageDraw.Draw(im)
    d.ellipse((u-r, v-r, u+r, v+r), outline=color, width=3)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="vis")
    p.add_argument("--num", type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    model_name = ckpt["model"]
    pretrained = ckpt.get("pretrained", False)
    mean, std = ckpt.get("norm", ((0.5,)*3, (0.5,)*3))
    model = build_model(model_name, pretrained=pretrained)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dataset = NoseDataset(args.data_root, "test-noses.txt", input_size=227, augment=None, normalize=(mean, std))

    idxs = random.sample(range(len(dataset)), args.num)
    for k, idx in enumerate(idxs, 1):
        tensor, target, meta = dataset[idx]
        img = (tensor.clone() * torch.tensor(std)[:,None,None] + torch.tensor(mean)[:,None,None]).clamp(0,1)
        img = (img*255).byte().permute(1,2,0).numpy()
        im = Image.fromarray(img)
        pred = model(tensor.unsqueeze(0)).squeeze(0).tolist()
        gt = target.tolist()
        draw_point(im, gt[0], gt[1], "lime")
        draw_point(im, pred[0], pred[1], "red")
        out_path = os.path.join(args.out_dir, f"vis_{k}_{meta['file']}")
        im.save(out_path)
        print("Saved", out_path)

if __name__ == "__main__":
    main()
