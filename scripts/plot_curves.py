# scripts/plot_curves.py
import os, argparse, json, csv, math, glob
import numpy as np
import matplotlib.pyplot as plt

def moving_avg(x, k=1):
    if k <= 1: 
        return np.array(x, dtype=float)
    x = np.array(x, dtype=float)
    pad = np.pad(x, (k-1, 0), mode='edge')
    w = np.ones(k) / k
    return np.convolve(pad, w, mode='valid')

def load_history(path):
    with open(path, 'r') as f:
        h = json.load(f)
    return {
        "file": path,
        "train": h.get("train_loss", []),
        "val":   h.get("val_loss", []),
        "lr":    h.get("lr", [])
    }

def plot_losses(histories, out_dir, smooth=1):
    if not histories: 
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    for H in histories:
        t = moving_avg(H["train"], smooth)
        v = moving_avg(H["val"],   smooth)
        epochs = np.arange(1, len(t)+1)
        plt.plot(epochs, t, label=os.path.basename(H["file"])+" — train")
        plt.plot(epochs, v, label=os.path.basename(H["file"])+" — val", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Train vs. validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print("Wrote", out)

def plot_lr(histories, out_dir):
    keep = [H for H in histories if H["lr"]]
    if not keep:
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    for H in keep:
        lr = H["lr"]
        epochs = np.arange(1, len(lr)+1)
        plt.plot(epochs, lr, label=os.path.basename(H["file"]))
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("LR schedule")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(out_dir, "lr_schedule.png")
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print("Wrote", out)

def load_errors_from_csv(path):
    errors = []
    with open(path, 'r', newline='') as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        for row in rdr:
            try:
                errors.append(float(row[-1]))
            except Exception:
                pass
    return np.array(errors, dtype=float)

def plot_error_hist_and_cdf(eval_csv, out_dir, bins=30):
    if not eval_csv: 
        return
    os.makedirs(out_dir, exist_ok=True)
    E = load_errors_from_csv(eval_csv)
    if E.size == 0:
        print("No errors parsed from", eval_csv)
        return
    plt.figure()
    plt.hist(E, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel("Localization error (pixels)")
    plt.ylabel("Count")
    plt.title(f"Error histogram — {os.path.basename(eval_csv)}\n"
              f"min={E.min():.2f}, mean={E.mean():.2f}, std={E.std():.2f}, max={E.max():.2f}")
    plt.grid(True, alpha=0.3)
    out1 = os.path.join(out_dir, "error_histogram.png")
    plt.savefig(out1, dpi=160, bbox_inches='tight')
    print("Wrote", out1)

    plt.figure()
    x = np.sort(E)
    y = np.arange(1, len(x)+1) / len(x)
    plt.plot(x, y)
    plt.xlabel("Localization error (pixels)")
    plt.ylabel("Cumulative fraction")
    plt.title("Error CDF")
    plt.grid(True, alpha=0.3)
    out2 = os.path.join(out_dir, "error_cdf.png")
    plt.savefig(out2, dpi=160, bbox_inches='tight')
    print("Wrote", out2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--histories", nargs="*", default=[], help="*_history.json files from training (glob allowed if your shell expands it)")
    ap.add_argument("--eval_csv", type=str, default=None, help="eval CSV from scripts/test.py")
    ap.add_argument("--out_dir", type=str, default="vis/plots")
    ap.add_argument("--smooth", type=int, default=1, help="moving-average window for loss curves")
    args = ap.parse_args()

    # Expand globs manually if a single arg contains wildcard
    hist_files = []
    for h in args.histories:
        if any(c in h for c in "*?[]"):
            hist_files.extend(glob.glob(h))
        else:
            hist_files.append(h)
    histories = [load_history(p) for p in hist_files if os.path.exists(p)]

    if histories:
        plot_losses(histories, args.out_dir, smooth=args.smooth)
        plot_lr(histories, args.out_dir)
    if args.eval_csv and os.path.exists(args.eval_csv):
        plot_error_hist_and_cdf(args.eval_csv, args.out_dir)

    if not histories and not (args.eval_csv and os.path.exists(args.eval_csv)):
        print("Nothing to plot. Provide --histories and/or --eval_csv.")

if __name__ == "__main__":
    main()
