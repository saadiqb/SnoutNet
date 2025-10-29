#!/usr/bin/env bash
set -euo pipefail

# ========== CONFIG ==========
DATA_ROOT="${1:-./data}"   # pass dataset dir as arg or keep default

EPOCHS=20

# LRs / batch sizes
SNOUT_LR=1e-3; SNOUT_BS=64
ALEX_LR=1e-4;  ALEX_BS=64
VGG_LR=1e-4;   VGG_BS=32

CKPT_DIR="checkpoints"
PLOT_DIR="vis"
LOG_DIR="logs"
mkdir -p "$CKPT_DIR" "$PLOT_DIR" "$LOG_DIR"

die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }
need python

# ---------- Split-file shim (supports underscores or hyphens) ----------
# If user has train_noses.txt / test_noses.txt, create hyphenated symlinks the code expects.
pushd "$DATA_ROOT" >/dev/null
if [[ -f train_noses.txt && ! -e train-noses.txt ]]; then ln -s train_noses.txt train-noses.txt; fi
if [[ -f test_noses.txt  && ! -e test-noses.txt  ]]; then ln -s test_noses.txt  test-noses.txt;  fi
[[ -f train-noses.txt ]] || die "Missing train-noses.txt (or train_noses.txt)"
[[ -f test-noses.txt  ]] || die "Missing test-noses.txt (or test_noses.txt)"
popd >/dev/null

echo "Using DATA_ROOT: $DATA_ROOT"
echo "Train split: $DATA_ROOT/train-noses.txt"
echo "Test  split: $DATA_ROOT/test-noses.txt"

log() { echo "[$(date +%H:%M:%S)] $*"; }

plot_losses() {
  python -m scripts.plot_curves --histories "$1" "$2" --out_dir "$3" --smooth 2 || true
}
plot_eval() {
  python -m scripts.plot_curves --eval_csv "$1" --out_dir "$2" || true
}
visualize() {
  python -m scripts.visualize --data_root "$DATA_ROOT" --ckpt "$1" --out_dir "$2" --num "${3:-6}" || true
}

# # ---------- 0) Reality check ----------
# log "Reality check (draw GT for a few train images)"
# python -m scripts.reality_check --data_root "$DATA_ROOT" 2>&1 | tee "$LOG_DIR/reality_check.log" || true

# # ---------- 1) SnoutNet (plain) ----------
# log "Train: SnoutNet (plain)"
# python -m scripts.train --data_root "$DATA_ROOT" --model snoutnet \
#   --epochs "$EPOCHS" --batch_size "$SNOUT_BS" --lr "$SNOUT_LR" \
#   --save "$CKPT_DIR" 2>&1 | tee "$LOG_DIR/snout_plain_train.log"

log "Test: SnoutNet (plain)"
python -m scripts.test --data_root "$DATA_ROOT" \
  --ckpt "$CKPT_DIR/snoutnet_plain.pt" --save_csv snoutnet_plain_eval.csv \
  2>&1 | tee "$LOG_DIR/snout_plain_test.log"

# plot_eval "snoutnet_plain_eval.csv" "$PLOT_DIR/plots_snout_plain"
# python -m scripts.plot_curves --histories "$CKPT_DIR/snoutnet_plain_history.json" --out_dir "$PLOT_DIR/plots_snout_plain"
# visualize "$CKPT_DIR/snoutnet_plain.pt" "$PLOT_DIR/snout_plain" 6

# # ---------- 2) SnoutNet (aug) ----------
# log "Train: SnoutNet (aug)"
# python -m scripts.train --data_root "$DATA_ROOT" --model snoutnet \
#   --epochs "$EPOCHS" --batch_size "$SNOUT_BS" --lr "$SNOUT_LR" \
#   --aug --save "$CKPT_DIR" 2>&1 | tee "$LOG_DIR/snout_aug_train.log"

log "Test: SnoutNet (aug)"
python -m scripts.test --data_root "$DATA_ROOT" \
  --ckpt "$CKPT_DIR/snoutnet_aug.pt" --save_csv snoutnet_aug_eval.csv \
  2>&1 | tee "$LOG_DIR/snout_aug_test.log"

# plot_losses "$CKPT_DIR/snoutnet_plain_history.json" "$CKPT_DIR/snoutnet_aug_history.json" "$PLOT_DIR/plots_snout_compare"
# plot_eval "snoutnet_aug_eval.csv" "$PLOT_DIR/plots_snout_aug"
# visualize "$CKPT_DIR/snoutnet_aug.pt" "$PLOT_DIR/snout_aug" 6

# # ---------- 3) AlexNet (plain) ----------
# log "Train: AlexNet (plain, pretrained)"
# python -m scripts.train --data_root "$DATA_ROOT" --model alexnet --pretrained \
#   --epochs "$EPOCHS" --batch_size "$ALEX_BS" --lr "$ALEX_LR" \
#   --save "$CKPT_DIR" 2>&1 | tee "$LOG_DIR/alex_plain_train.log"

log "Test: AlexNet (plain)"
python -m scripts.test --data_root "$DATA_ROOT" \
  --ckpt "$CKPT_DIR/alexnet_plain.pt" --save_csv alexnet_plain_eval.csv \
  2>&1 | tee "$LOG_DIR/alex_plain_test.log"

# python -m scripts.plot_curves --histories "$CKPT_DIR/alexnet_plain_history.json" --out_dir "$PLOT_DIR/plots_alex"
# plot_eval "alexnet_plain_eval.csv" "$PLOT_DIR/plots_alex"
# visualize "$CKPT_DIR/alexnet_plain.pt" "$PLOT_DIR/alex_plain" 4

# # ---------- 4) AlexNet (aug) ----------
# log "Train: AlexNet (aug, pretrained)"
# python -m scripts.train --data_root "$DATA_ROOT" --model alexnet --pretrained \
#   --epochs "$EPOCHS" --batch_size "$ALEX_BS" --lr "$ALEX_LR" \
#   --aug --save "$CKPT_DIR" 2>&1 | tee "$LOG_DIR/alex_aug_train.log"

log "Test: AlexNet (aug)"
python -m scripts.test --data_root "$DATA_ROOT" \
  --ckpt "$CKPT_DIR/alexnet_aug.pt" --save_csv alexnet_aug_eval.csv \
  2>&1 | tee "$LOG_DIR/alex_aug_test.log"

# python -m scripts.plot_curves --histories "$CKPT_DIR/alexnet_plain_history.json" "$CKPT_DIR/alexnet_aug_history.json" --out_dir "$PLOT_DIR/plots_alex" --smooth 2
# plot_eval "alexnet_aug_eval.csv" "$PLOT_DIR/plots_alex"
# visualize "$CKPT_DIR/alexnet_aug.pt" "$PLOT_DIR/alex_aug" 4

# # ---------- 5) VGG16 (plain) ----------
# log "Train: VGG16 (plain, pretrained)"
# python -m scripts.train --data_root "$DATA_ROOT" --model vgg16 --pretrained \
#   --epochs "$EPOCHS" --batch_size "$VGG_BS" --lr "$VGG_LR" \
#   --save "$CKPT_DIR" 2>&1 | tee "$LOG_DIR/vgg_plain_train.log"

log "Test: VGG16 (plain)"
python -m scripts.test --data_root "$DATA_ROOT" \
  --ckpt "$CKPT_DIR/vgg16_plain.pt" --save_csv vgg16_plain_eval.csv \
  2>&1 | tee "$LOG_DIR/vgg_plain_test.log"

# python -m scripts.plot_curves --histories "$CKPT_DIR/vgg16_plain_history.json" --out_dir "$PLOT_DIR/plots_vgg"
# plot_eval "vgg16_plain_eval.csv" "$PLOT_DIR/plots_vgg"
# visualize "$CKPT_DIR/vgg16_plain.pt" "$PLOT_DIR/vgg_plain" 4

# # ---------- 6) VGG16 (aug) ----------
# log "Train: VGG16 (aug, pretrained)"
# python -m scripts.train --data_root "$DATA_ROOT" --model vgg16 --pretrained \
#   --epochs "$EPOCHS" --batch_size "$VGG_BS" --lr "$VGG_LR" \
#   --aug --save "$CKPT_DIR" 2>&1 | tee "$LOG_DIR/vgg_aug_train.log"

log "Test: VGG16 (aug)"
python -m scripts.test --data_root "$DATA_ROOT" \
  --ckpt "$CKPT_DIR/vgg16_aug.pt" --save_csv vgg16_aug_eval.csv \
  2>&1 | tee "$LOG_DIR/vgg_aug_test.log"

# python -m scripts.plot_curves --histories "$CKPT_DIR/vgg16_plain_history.json" "$CKPT_DIR/vgg16_aug_history.json" --out_dir "$PLOT_DIR/plots_vgg" --smooth 2
# plot_eval "vgg16_aug_eval.csv" "$PLOT_DIR/plots_vgg"
# visualize "$CKPT_DIR/vgg16_aug.pt" "$PLOT_DIR/vgg_aug" 4

# ---------- 7) Ensemble ----------
log "Ensemble (SnoutNet + AlexNet + VGG16, augmented)"
python -m scripts.ensemble --data_root "$DATA_ROOT" \
  --snoutnet_ckpt "$CKPT_DIR/snoutnet_aug.pt" \
  --alexnet_ckpt  "$CKPT_DIR/alexnet_aug.pt" \
  --vgg16_ckpt    "$CKPT_DIR/vgg16_aug.pt" \
  2>&1 | tee "$LOG_DIR/ensemble.log"

log "All done. Check '$CKPT_DIR', '$PLOT_DIR', and '$LOG_DIR'."
