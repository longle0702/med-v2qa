# Medical VQA — MUMC

Implementation of **MUMC** (Masked Vision and Language Pre-training with Unimodal and Multimodal Contrastive Losses) for Medical Visual Question Answering, accepted at [MICCAI 2023](https://conferences.miccai.org/2023/en/).

> Based on [MUMC](https://github.com/pengfeiliHEU/MUMC) by Pengfei Li et al. and inspired by [ALBEF](https://github.com/salesforce/ALBEF).

---

## Pretrained Weights

The fine-tuned VQA checkpoint (epoch 31, ~2.27 GB) is hosted on Hugging Face and **cannot** be stored in this repo due to GitHub LFS limits:

🤗 **[longle0702/medical-vqa on Hugging Face](https://huggingface.co/longle0702/medical-vqa)**

```bash
# Download with huggingface_hub
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="longle0702/medical-vqa", filename="med_pretrain_29_rad_31.pth", local_dir="./output/rad/")
```

The pre-training checkpoint (`pretrain/med_pretrain_29.pth`) is tracked via Git LFS in this repo.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
```

---

## Training & Evaluation

```bash
# Fine-tune on VQA-RAD
python train_vqa.py --dataset_use rad --checkpoint ./pretrain/med_pretrain_29.pth --output_dir ./output/rad

# Evaluate
python eval.py

# Semantic evaluation (requires GEMINI_API_KEY in .env)
python semantic_eval.py
```

---

## Citation

```bibtex
@article{MUMC,
  title     = {Masked Vision and Language Pre-training with Unimodal and Multimodal Contrastive Losses for Medical Visual Question Answering},
  author    = {Pengfei Li, Gang Liu, Jinlong He, Zixu Zhao and Shenjun Zhong},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023},
  year      = {2023},
  pages     = {374--383},
  publisher = {Springer Nature Switzerland}
}
```

## License

MIT License
