# Med-V²QA: Clinical Inference Engine

This project provides a robust, production-ready Visual Question Answering (VQA) inference engine for medical imaging. It builds upon the **MUMC** architecture ([MICCAI 2023](https://conferences.miccai.org/2023/en/)) and extends it with a comprehensive API, clinical safety guardrails, batch triage, audio readout, multilingual support, and PDF report generation.

> **Architecture credit:** [MUMC by Pengfei Li et al.](https://github.com/pengfeiliHEU/MUMC), inspired by ALBEF.

## Key Features

- **MUMC Medical VQA:** Highly accurate answers to clinical questions about radiology images.
- **Dual-Gate Guardrail System:** Ensures clinical safety and system focus:
  - *Gate 1 (Intent):* Rejects non-medical or conversational questions (e.g., "tell me a joke").
  - *Gate 2 (Confidence):* Rejects non-medical images (e.g., selfies, pets) using a local CLIP classifier.
- **Batch Triage:** Upload multiple scans (e.g., up to 20 images) to automatically sort and prioritise the most abnormal scans first using zero-shot CLIP.
- **Voice I/O & Accessibility:** 
  - **Speech-to-Text:** Built-in `openai/whisper-base.en` transcription runs locally to turn spoken questions into text.
  - **Text-to-Speech:** Integrated audio readout of answers using `gTTS` with adjustable playback speed.
- **Advanced Clinical Web UI:** A bundled, single-page frontend (`frontend/index.html`) offering:
  - **Multilingual Support:** English (EN), French (FR), and Vietnamese (VI) interfaces.
  - **PDF Reports:** Generate and download structured clinical reports with patient data, notes, and the VQA results.
  - **Interactive Interface:** Drag-and-drop for images and triage batches, voice recording toggles.
- **Production Deployment:**
  - **FastAPI** backend for high-performance async request handling.
  - **Nginx & Systemd:** Production-ready configuration for reverse proxying and auto-starting the service.
- **Single-Load Architecture:** The 2.2GB MUMC checkpoint is loaded exactly once into a shared `InferenceEngine` singleton used across all API services.

---

## Quickstart (Docker Recommended)

The easiest way to run the full application is using Docker Compose.

### 1. Download Required Models

You'll need both the VQA checkpoint and the CLIP guardrail model. They are too large for Git and must be downloaded manually.

**MUMC VQA Checkpoint:**
```bash
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="longle0702/medical-vqa", filename="med_pretrain_29_rad_34.pth", local_dir=".")
```

**CLIP Guardrail Model:**
```bash
import os
from transformers import CLIPModel, CLIPProcessor
CLIP_MODEL_PATH = os.path.join(os.getcwd(), "guardrail", "clip_model")

# Transformers < 4.27 defaults to PyTorch bins; explicitly save in PyTorch format for compatibility
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="openai/clip-vit-base-patch32", filename="pytorch_model.bin", local_dir=CLIP_MODEL_PATH)
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").save_pretrained(CLIP_MODEL_PATH)
```

### 2. Start the Server

```bash
docker compose up -d --build
```

The API will be available at `http://localhost:8000`.
- **Clinical UI:** `http://localhost:8000/`
- **Health Check:** `http://localhost:8000/health`

---

## Manual Setup & Production Deployment

If you prefer to run the API without Docker or want to deploy to production:

### Local Development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the models (as detailed in step 1 above)

# 3. Start the FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment (Nginx + Systemd)
The repository includes configuration for a production Linux environment (e.g., Ubuntu).
1. Copy the systemd service file: `sudo cp systemd/medv2qa.service /etc/systemd/system/`
2. Start the service: `sudo systemctl enable --now medv2qa`
3. Copy the Nginx config: `sudo cp nginx/medv2qa.mm.net.vn /etc/nginx/sites-available/`
4. Enable the site: `sudo ln -s /etc/nginx/sites-available/medv2qa.mm.net.vn /etc/nginx/sites-enabled/`
5. Reload Nginx: `sudo systemctl reload nginx`

---

## Training & Evaluation (Original MUMC Pipeline)

The original training and evaluation scripts have been preserved and enhanced with semantic scoring (Gemini).

```bash
# Fine-tune on VQA-RAD (requires pretrain weights in ./pretrain/med_pretrain_29.pth)
python train_vqa.py --dataset_use rad --checkpoint ./pretrain/med_pretrain_29.pth --output_dir ./output/rad

# Evaluate standard metrics
python eval.py

# Semantic evaluation (requires GEMINI_API_KEY in .env)
cp .env.example .env  # fill in GEMINI_API_KEY
python semantic_eval.py
```

---

## Citation

If you use this architecture, please cite the original MUMC paper:

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
