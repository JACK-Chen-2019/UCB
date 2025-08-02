# UCB: Enhancing Continual Semantic Segmentation via Uncertainty and Class Balance Re-weighting


🔗 Official Jittor implementation of our TIP 2025 paper:
**["Enhancing Continual Semantic Segmentation via Uncertainty and Class Balance Re-weighting"](https://ieeexplore.ieee.org/document/11030217)**

---

## 🧩 Jittor Installation

```bash
python -m pip install jittor
```

---

## 📁 Dataset Preparation

Please follow the instructions below to prepare the datasets:

```bash
bash UCB_PLOP/data/download_voc.sh
bash UCB_PLOP/data/download_ade.sh
```

---

## 🚀 Training

To start training on VOC 15-1 setting with a single GPU, run:

```bash
bash scripts/voc/resnet101_15-1_singleGPU.sh
```

> Add the `--ucb` flag to enable UCB re-weighting.

## 🙏 Acknowledgements

This work builds upon the open-source implementations of:

- [PLOP (CVPR 2021)](https://github.com/arthurdouillard/CVPR2021_PLOP)
- [NeST (ECCV 2024)](https://github.com/zhengyuan-xie/ECCV24_NeST)