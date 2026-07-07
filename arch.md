# Med-V²QA — Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      DEPLOYMENT LAYER                                       │
│                                                                                             │
│   docker-compose.yml                                                                        │
│   ┌─────────────────────────────────────────────────────────────────┐                      │
│   │  Container: med-v2qa-api   (port 8000:8000)                     │                      │
│   │  Image: med-v2qa-api:latest  (Dockerfile)                       │                      │
│   │  Volume (read-only): jmmkkmjmmjmmmv ./med_pretrain_29_rad_34.pth (2.2 GB)      │                      │
│   │  Healthcheck: GET /health every 30 s                            │                      │
│   └─────────────────────────────────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    CLIENT / BROWSER                                         │
│                                                                                             │
│   frontend/index.html  (single-page clinical UI — served as static by FastAPI)             │
│                                                                                             │
│   ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────────────┐  │
│   │   VQA Panel          │   │   Triage Panel        │   │   Voice Controls             │  │
│   │  · Upload image      │   │  · Upload batch of    │   │  · MediaRecorder (WebM/Opus) │  │
│   │  · Type question     │   │    images (<=50)      │   │  · POST /transcribe          │  │
│   │  · POST /predict     │   │  · POST /triage       │   │  · POST /readout (TTS)       │  │
│   └──────────┬───────────┘   └──────────┬────────────┘   └──────────────────────────────┘  │
└──────────────┼──────────────────────────┼──────────────────────────────────────────────────┘
               │  HTTP (multipart/form)   │  HTTP (multipart/form)
               v                          v
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                          FastAPI APPLICATION  (api/main.py)                                 │
│                                                                                             │
│   CORS Middleware (allow_origins=["*"])                                                     │
│   Static files: /static  ->  frontend/                                                     │
│                                                                                             │
│  ┌────────────┐  ┌────────────────────────┐  ┌───────────────────┐  ┌───────────────────┐ │
│  │GET /health │  │  POST /predict         │  │  POST /triage     │  │ POST /transcribe  │ │
│  │            │  │  (guarded single-img   │  │  (batch abnorm-   │  │ POST /readout     │ │
│  │ -> backend │  │   VQA inference)       │  │   ality sorting)  │  │ (voice I/O)       │ │
│  │   device   │  └──────────┬─────────────┘  └────────┬──────────┘  └────────┬──────────┘ │
│  └────────────┘             │                          │                      │            │
│                                                                                             │
│   -- Startup / lifespan -----------------------------------------------------------         │
│   1. InferenceEngine()      <- loads MUMC .pth checkpoint once                             │
│   2. GuardrailPipeline(preloaded_model=engine.model, ...)                                  │
│   3. BatchTriageService(device=engine.device)                                              │
│   4. Whisper (openai/whisper-base.en)                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
               │                          │                      │
               v                          │                      v
┌──────────────────────────────────────┐  │   ┌────────────────────────────────────────────┐
│  GUARDRAIL PIPELINE                  │  │   │  VOICE PIPELINE                            │
│  guardrail/pipeline.py               │  │   │                                            │
│                                      │  │   │  Speech-to-Text: Whisper (base.en)         │
│  ┌────────────────────────────────┐  │  │   │  · librosa -> 16 kHz mono waveform         │
│  │ Gate 1: Text Intent            │  │  │   │  · AutoProcessor + AutoModelForSpeechSeq2Seq│
│  │ guardrail/intent_classifier.py │  │  │   │  · returns TranscribeResponse              │
│  │                                │  │  │   │                                            │
│  │ · Feeds blank (zero) image to  │  │  │   │  Text-to-Speech: gTTS                      │
│  │   MUMC visual encoder          │  │  │   │  · returns MP3 audio/mpeg stream           │
│  │ · Runs question through text   │  │  │   └────────────────────────────────────────────┘
│  │   encoder + single decoder     │  │
│  │   step (BOS -> first token)    │  │
│  │ · Sums prob over MEDICAL_SEED_ │  │
│  │   TOKENS (anatomy/radiology)   │  │
│  │ · threshold check -> pass/fail │  │
│  └──────────────┬─────────────────┘  │
│                 │ pass               │
│                 v                    │
│  ┌────────────────────────────────┐  │
│  │ Gate 2: Image Confidence       │  │
│  │ guardrail/confidence_gate.py   │  │
│  │                                │  │
│  │ · CLIP (openai/clip-vit-base-  │  │
│  │   patch32) loaded from local   │  │
│  │   cache (guardrail/clip_model/)│  │
│  │ · Zero-shot classification:    │  │
│  │   10 medical prompts vs        │  │
│  │   10 non-medical prompts       │  │
│  │ · Softmax medical_score >=     │  │
│  │   CLIP_THRESHOLD -> pass/fail  │  │
│  └──────────────┬─────────────────┘  │
│                 │ both pass          │
│                 v                    │
│  ┌────────────────────────────────┐  │
│  │ VQA Inference (predict)        │  │
│  │ inference.py                   │  │
│  │ api/engine.py -> InferenceEngine│ │
│  │                                │  │
│  │ 1. image -> visual_encoder     │  │
│  │    -> image_embeds [1, N, 768] │  │
│  │ 2. question -> text_encoder    │  │
│  │    (cross-attn on image_embeds)│  │
│  │    -> question_hidden_state    │  │
│  │ 3. beam-search decode          │  │
│  │    text_decoder.generate()     │  │
│  │    -> answer string            │  │
│  └────────────────────────────────┘  │
│                                      │
│  returns: GuardrailResult            │
│   .passed / .gate_triggered          │
│   .answer / .refusal_message         │
│   .intent_result                     │
│   .confidence_result                 │
│   .metadata (per-stage latencies)    │
└──────────────────────────────────────┘
               │
               v
┌──────────────────────────────────────────────────────────┐
│  MUMC VQA MODEL  (models/)                               │
│                                                          │
│  models/model_vqa.py  ->  MUMC_VQA (nn.Module)          │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Visual Encoder                                     │  │
│  │ models/vision/vit.py  ->  VisionTransformer        │  │
│  │  img_size=480, patch_size=16, embed_dim=768        │  │
│  │  depth=12, num_heads=12                            │  │
│  │  (MAE variants: vision/mae.py, vision/mae_v2.py)  │  │
│  │  (patch embeds: vision/path_embeds.py)             │  │
│  └───────────────────────────┬────────────────────────┘  │
│                              │ image_embeds [1, N, 768]  │
│  ┌───────────────────────────v────────────────────────┐  │
│  │ Text Encoder                                       │  │
│  │ models/xbert.py  ->  BertModel (cross-attention)   │  │
│  │ models/tokenization_bert.py  ->  BertTokenizer     │  │
│  │  pre-trained: bert-base-uncased                    │  │
│  │  cross-attends image_embeds + question tokens      │  │
│  └───────────────────────────┬────────────────────────┘  │
│                              │ question_hidden_state      │
│  ┌───────────────────────────v────────────────────────┐  │
│  │ Text Decoder                                       │  │
│  │ models/xbert.py  ->  BertLMHeadModel               │  │
│  │  6 layers, hidden_size=720                         │  │
│  │  beam-search generate() -> answer tokens           │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Checkpoint: med_pretrain_29_rad_34.pth  (2.2 GB)        │
│  Config:     configs/VQA.yaml                            │
│              configs/config_bert.json                    │
│  Optional: distillation momentum encoder pairs           │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  BATCH TRIAGE SERVICE  (triage/)                         │
│                                                          │
│  triage/batch_sorter.py  ->  BatchTriageService          │
│                                                          │
│  · Independent CLIP model (same local cache as Gate 2)   │
│  · Inputs: list of image paths (<=50)                    │
│  · Batched CLIP inference:                               │
│    - 4 abnormal prompts  vs  4 normal prompts            │
│    - logits_per_image -> softmax -> abnormality score    │
│  · Sort descending by score (most critical first)        │
│  · Returns: [{ image_path, score, is_abnormal }, ...]    │
│                                                          │
│  Also uses guardrail.check_image() to pre-filter         │
│  non-medical images before scoring                        │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  TRAINING & EVALUATION SCRIPTS  (scripts/)               │
│                                                          │
│  scripts/train_vqa.py    -- VQA fine-tuning loop         │
│  scripts/eval.py         -- evaluation harness           │
│  scripts/vqaEvaluate.py  -- accuracy metrics             │
│                                                          │
│  dataset/                                                │
│  +-- __init__.py          dataset factory                │
│  +-- vqa_dataset.py       VQA sample loader              │
│  +-- pretrain_dataset.py  pre-training data loader       │
│  +-- randaugment.py       data augmentation              │
│  +-- utils.py             pre_question(), tokenize utils │
│                                                          │
│  vqaTools/                                               │
│  +-- vqa.py               VQA annotation loader (COCO)   │
│  +-- vqaEval.py           official VQA accuracy eval     │
│                                                          │
│  data_RAD/                VQA-RAD dataset (JSON/images)  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  CONFIGURATION  (configs/)                               │
│                                                          │
│  VQA.yaml         image_res, batch sizes, lr, etc.       │
│  Pretrain.yaml    pre-training hyperparameters           │
│  config_bert.json BERT encoder/decoder architecture      │
└──────────────────────────────────────────────────────────┘
```

---

## Data / Request Flow Summary

```
User (Browser)
    |
    |  POST /predict  { image, question }
    v
FastAPI (api/main.py)
    |
    |  writes image bytes -> temp file
    v
GuardrailPipeline.run()
    |
    +--[Gate 1: Text Intent]---------------------------------------------+
    |   MedicalIntentClassifier                                          |
    |   blank image -> visual_encoder -> text_encoder ->                 |
    |   single decoder step -> seed-token softmax sum                    |
    |   score < threshold  -->  REFUSAL (intent)                        |
    |   score >= threshold -->  continue --------------------------------+
    |
    +--[Gate 2: Image Confidence]----------------------------------------+
    |   ConfidenceGate (CLIP)                                            |
    |   image vs. medical/non-medical prompts                            |
    |   medical_score < threshold  -->  REFUSAL (confidence)            |
    |   medical_score >= threshold -->  continue +-----------------------+
    |
    +--[VQA Inference]
        image -> ViT visual_encoder -> image_embeds
        question -> BERT text_encoder (cross-attn) -> hidden_state
        BOS -> BERT text_decoder beam-search -> answer string
        --> PredictResponse { passed, answer, gate_detail, latency_ms }


User (Browser)
    |
    |  POST /triage  { images[] }
    v
FastAPI
    |  for each image: guardrail.check_image() -> gate 2 screen
    |  accepted images -> BatchTriageService.sort_batch()
    |      CLIP zero-shot abnormality scoring (batched)
    |      sort descending by score
    --> TriageResponse { queue: [{ filename, score, is_abnormal }] }


User (Browser)
    |
    |  POST /transcribe  { audio (WebM/Opus) }
    v
FastAPI -> librosa -> 16 kHz mono -> Whisper -> transcript string


User (Browser)
    |
    |  POST /readout  { text }
    v
FastAPI -> gTTS -> MP3 audio stream
```

---

## Module Dependency Map

```
api/main.py
 +-- api/engine.py           (InferenceEngine: loads .pth, exposes model/tokenizer/transform)
 |    +-- inference.py       (load_model, predict)
 |         +-- models/model_vqa.py          (MUMC_VQA)
 |         |    +-- models/vision/vit.py    (VisionTransformer)
 |         |    +-- models/vision/mae.py    (MAE variants)
 |         |    +-- models/vision/mae_v2.py
 |         |    +-- models/vision/path_embeds.py  (patch embeddings)
 |         |    +-- models/xbert.py         (BertModel, BertLMHeadModel)
 |         +-- models/tokenization_bert.py  (BertTokenizer)
 |         +-- dataset/utils.py             (pre_question)
 |
 +-- api/schemas.py          (Pydantic response models)
 |
 +-- guardrail/pipeline.py   (GuardrailPipeline)
 |    +-- guardrail/intent_classifier.py    (MedicalIntentClassifier, reuses MUMC model)
 |    +-- guardrail/confidence_gate.py      (ConfidenceGate, CLIP-based)
 |    |    +-- guardrail/clip_model/        (local CLIP weights cache)
 |    +-- guardrail/refusal.py              (build_refusal)
 |    +-- guardrail/result_types.py         (GuardrailResult, IntentResult, ConfidenceResult)
 |    +-- guardrail/config.py               (thresholds, paths, MEDICAL_SEED_TOKENS)
 |
 +-- triage/batch_sorter.py  (BatchTriageService: CLIP abnormality scoring)

configs/
 +-- VQA.yaml
 +-- Pretrain.yaml
 +-- config_bert.json

scripts/
 +-- train_vqa.py            (fine-tuning)
 +-- eval.py                 (evaluation)
 +-- vqaEvaluate.py

dataset/
 +-- vqa_dataset.py
 +-- pretrain_dataset.py
 +-- randaugment.py
 +-- utils.py

vqaTools/
 +-- vqa.py
 +-- vqaEval.py

frontend/
 +-- index.html              (single-file clinical UI)
```
