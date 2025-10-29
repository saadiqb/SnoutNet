# SnoutNet — Project README

Pet Nose Localization

Models (implementation files)
- [models/snoutnet.py](models/snoutnet.py)
- [models/snoutnet_alexnet.py](models/snoutnet_alexnet.py)
- [models/snoutnet_vgg.py](models/snoutnet_vgg.py)

Scripts (entry points)
- [`scripts/train.py`](scripts/train.py) — main training entry
- [`scripts/test.py`](scripts/test.py) — evaluation / quick checks
- [`scripts/ensemble.py`](scripts/ensemble.py) — ensemble builder/evaluator
- [`scripts/compare_models.py`](scripts/compare_models.py) — compare multiple runs
- [`scripts/plot_curves.py`](scripts/plot_curves.py) — generate training/validation plots
- [`scripts/reality_check.py`](scripts/reality_check.py) — sanity checks / baseline tests
- [`scripts/visualize.py`](scripts/visualize.py) — produce sample visualizations
- [`scripts/test.py`](scripts/test.py)

Getting started (example)
1. Create virtual environment and install requirements:
   - pip install -r [requirements.txt](requirements.txt)
2. Prepare data:
   - Place raw or preprocessed data into [data/](data/)
3. Train a model (example):
   - python [scripts/train.py](scripts/train.py) --config <your-config>
4. Run evaluation:
   - python [scripts/test.py](scripts/test.py) --model <path-to-checkpoint>
5. Visualize results:
   - python [scripts/visualize.py](scripts/visualize.py) --run <run-id>
6. Generate summary plots:
   - python [scripts/plot_curves.py](scripts/plot_curves.py) --input-dir logs/