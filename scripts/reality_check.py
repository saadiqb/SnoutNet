
# scripts/reality_check.py
import os, argparse
from PIL import Image, ImageDraw
from dataset import NoseDataset

def draw_point(im: Image.Image, u: float, v: float, color, r=5):
    d = ImageDraw.Draw(im)
    d.ellipse((u-r, v-r, u+r, v+r), outline=color, width=3)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="vis/reality_check")
    p.add_argument("--num", type=int, default=8)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = NoseDataset(args.data_root, "train-noses.txt", input_size=227, augment=None)
    print("Parsed", len(ds), "samples")
    for i in range(min(args.num, len(ds))):
        tensor, target, meta = ds[i]
        im = Image.fromarray((tensor.permute(1,2,0).numpy()*255).astype("uint8"))
        u, v = target.tolist()
        draw_point(im, u, v, "cyan", r=4)
        im.save(os.path.join(args.out_dir, f"rc_{i:03d}_{meta['file']}"))
        print(meta["file"], target.tolist())

if __name__ == "__main__":
    main()
