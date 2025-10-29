# scripts/compare_models.py  (legend-fixed)
import os, argparse, json, csv
from typing import List, Tuple
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

from utils import load_checkpoint, summarize_errors
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

def draw_point(im: Image.Image, u: float, v: float, color, r=5, w=3):
    d = ImageDraw.Draw(im)
    d.ellipse((u-r, v-r, u+r, v+r), outline=color, width=w)

def parse_split_file(split_path: str) -> List[Tuple[str, Tuple[int, int]]]:
    out = []
    with open(split_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname, rest = line.split(",", 1)
            rest = rest.strip().strip('"').strip().strip("()")
            u_str, v_str = rest.split(",")
            u, v = int(u_str.strip()), int(v_str.strip())
            out.append((fname, (u, v)))
    if not out:
        raise RuntimeError(f"No samples parsed from {split_path}")
    return out

def load_and_prepare_image(img_path: str, target_uv, input_size: int, mean, std):
    im = Image.open(img_path).convert("RGB")
    w0, h0 = im.size
    im_resized = TF.resize(im, [input_size, input_size])
    w1 = h1 = input_size
    su, sv = (w1 / w0), (h1 / h0)
    u_scaled = target_uv[0] * su
    v_scaled = target_uv[1] * sv
    t = TF.to_tensor(im_resized)
    t = TF.normalize(t, mean, std)
    return t, (u_scaled, v_scaled), im_resized.copy()

@torch.no_grad()
def forward_uv(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.unsqueeze(0).to(device)
    y = model(x).squeeze(0).cpu()
    return y

def textsize(draw: ImageDraw.ImageDraw, text: str, font):
    # robust size (textbbox works in modern Pillow)
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        return bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        return draw.textsize(text, font=font)

def render_with_legend(base_img: Image.Image,
                       fname: str,
                       model_names: List[str],
                       colors: List[str],
                       errs: List[float],
                       title_font=None, label_font=None):
    # Fonts
    if title_font is None:
        try:
            title_font = ImageFont.load_default()
        except:
            title_font = None
    if label_font is None:
        label_font = title_font

    # Measure legend rows
    legend_rows = [("GT", "lime", None)] + list(zip(model_names, colors, errs))
    draw_tmp = ImageDraw.Draw(base_img)
    row_h = 0
    swatch_w, swatch_h = 24, 24
    pad_x, pad_y = 12, 10
    gap = 14

    # compute max text width
    max_text_w = 0
    for name, _, err in legend_rows:
        txt = name if err is None else f"{name}  (err={err:.1f}px)"
        tw, th = textsize(draw_tmp, txt, label_font)
        max_text_w = max(max_text_w, tw)
        row_h = max(row_h, max(th, swatch_h))

    # Legend layout: two rows (title+filename) + all entries in a single column
    title = "Model comparison"
    subtitle = fname
    title_w, title_h = textsize(draw_tmp, title, title_font)
    sub_w, sub_h = textsize(draw_tmp, subtitle, label_font)

    legend_w = pad_x*2 + swatch_w + 8 + max_text_w
    legend_h = pad_y*3 + title_h + sub_h + gap + len(legend_rows)* (row_h + 8)

    canvas = Image.new("RGB", (base_img.width, base_img.height + legend_h), (255,255,255))
    canvas.paste(base_img, (0,0))
    d = ImageDraw.Draw(canvas)

    # Title + subtitle
    x = pad_x; y = base_img.height + pad_y
    d.text((x, y), title, fill=(0,0,0), font=title_font); y += title_h + 2
    d.text((x, y), subtitle, fill=(40,40,40), font=label_font); y += sub_h + gap

    # Rows
    for name, color, err in legend_rows:
        # swatch
        d.rectangle([x, y + (row_h - swatch_h)//2, x + swatch_w, y + (row_h + swatch_h)//2],
                    outline=color, width=3)
        # label
        txt = name if err is None else f"{name}  (err={err:.1f}px)"
        d.text((x + swatch_w + 8, y + (row_h - sub_h)//2), txt, fill=(0,0,0), font=label_font)
        y += row_h + 8

    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt1", required=True)
    ap.add_argument("--ckpt2", required=True)
    ap.add_argument("--ckpt3", required=True)
    ap.add_argument("--ckpt4", required=True)
    ap.add_argument("--test_file", default="test-noses.txt")
    ap.add_argument("--input_size", type=int, default=227)
    ap.add_argument("--out_dir", default="vis/compare4")
    ap.add_argument("--max_vis", type=int, default=12)
    ap.add_argument("--save_csv", default=None)
    ap.add_argument("--colors", nargs=4, default=["red","deepskyblue","magenta","gold"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoints
    ckpt_paths = [args.ckpt1, args.ckpt2, args.ckpt3, args.ckpt4]
    ckpts = [load_checkpoint(p, map_location="cpu") for p in ckpt_paths]

    def tag_from_ckpt(ck):
        tag = ck.get("model","model")
        tag += "_aug" if ck.get("aug", False) else "_plain"
        return tag

    names = [tag_from_ckpt(ck) for ck in ckpts]
    norms = [ck.get("norm", ((0.5,)*3, (0.5,)*3)) for ck in ckpts]

    # Build models
    def mk(ck):
        mname = ck["model"]; pre = ck.get("pretrained", False)
        if mname == "snoutnet": m = SnoutNet()
        elif mname == "alexnet": m = SnoutNetAlexNet(pretrained=pre)
        elif mname == "vgg16":  m = SnoutNetVGG16(pretrained=pre)
        else: raise ValueError(mname)
        m.load_state_dict(ck["state_dict"]); m.to(device).eval(); return m
    models = [mk(ck) for ck in ckpts]

    # Split list
    test_list = parse_split_file(os.path.join(args.data_root, args.test_file))

    # Evaluation accumulators
    all_errs = [[] for _ in range(4)]
    if args.save_csv:
        rows = [("file","gt_u","gt_v",
                 f"{names[0]}_u", f"{names[0]}_v", f"{names[0]}_err",
                 f"{names[1]}_u", f"{names[1]}_v", f"{names[1]}_err",
                 f"{names[2]}_u", f"{names[2]}_v", f"{names[2]}_err",
                 f"{names[3]}_u", f"{names[3]}_v", f"{names[3]}_err")]

    vis_done = 0
    for fname, (u0, v0) in test_list:
        img_path = os.path.join(args.data_root, "images-original", "images", fname)

        # Prepare inputs per model (own mean/std)
        tensors = []; scaled_uv = None; disp_img = None
        for k in range(4):
            t, scaled_uv, disp_img = load_and_prepare_image(img_path, (u0, v0), args.input_size, *norms[k])
            tensors.append(t)

        # Predict
        gt = torch.tensor(scaled_uv, dtype=torch.float32)
        preds = []; errs = []
        for k in range(4):
            y = forward_uv(models[k], tensors[k], device)
            preds.append(y)
            errs.append(float(torch.linalg.vector_norm(y - gt)))

        for k in range(4):
            all_errs[k].append(errs[k])

        if args.save_csv:
            rows.append((fname, scaled_uv[0], scaled_uv[1],
                         preds[0][0].item(), preds[0][1].item(), errs[0],
                         preds[1][0].item(), preds[1][1].item(), errs[1],
                         preds[2][0].item(), preds[2][1].item(), errs[2],
                         preds[3][0].item(), preds[3][1].item(), errs[3]))

        # Visualize a few
        if vis_done < args.max_vis:
            im = disp_img.copy()
            # GT
            draw_point(im, scaled_uv[0], scaled_uv[1], "lime", r=6, w=4)
            # Predictions
            for k, color in enumerate(args.colors):
                draw_point(im, preds[k][0].item(), preds[k][1].item(), color, r=5, w=3)

            # Render big, explicit legend with filename + per-model errors
            panel = render_with_legend(im, fname, names, args.colors, errs)
            out_path = os.path.join(args.out_dir, f"cmp_{vis_done:03d}_{os.path.splitext(fname)[0]}.png")
            panel.save(out_path)
            print("Saved", out_path)
            vis_done += 1

    # Print stats
    for k in range(4):
        E = torch.tensor(all_errs[k], dtype=torch.float32)
        stats = summarize_errors(E)
        print(f"[{names[k]}] Evaluation (pixels):", json.dumps(stats, indent=2))

    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print("Wrote", args.save_csv)

if __name__ == "__main__":
    main()
