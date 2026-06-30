"""
guardrail/config.py
--------------------
Centralised configuration for the Dual-Gate Medical Guardrail.
All values can be overridden at ``GuardrailPipeline`` construction time;
these serve as sensible, tested defaults.
"""

from __future__ import annotations

import os

MEDICAL_SEED_TOKENS: list[str] = [
    # Radiology findings
    "abnormal", "normal", "opacity", "consolidation", "effusion",
    "pneumonia", "cardiomegaly", "atelectasis", "edema", "infiltrate",
    "pneumothorax", "nodule", "mass", "fracture", "lesion",
    # Anatomy
    "lung", "lungs", "chest", "heart", "liver", "kidney", "spine",
    "pleural", "bilateral", "unilateral", "mediastinum", "diaphragm",
    # Clinical terms
    "enlarged", "increased", "decreased", "present", "absent",
    "mild", "moderate", "severe", "acute", "chronic",
    "calcification", "density", "opaque", "lucent",
    # Simple / Layman Terms (What patients actually ask about)
    "pain", "hurt", "broken", "break", "spot", "shadow", "bleed",
    "bleeding", "swelling", "swollen", "lump", "bump", "hole", "tear",
    "fluid", "blood", "cough", "ache", "injury", "sick", "wrong",
    # Simple Anatomy (Basic body parts)
    "bone", "bones", "rib", "ribs", "neck", "head", "brain", "skull",
    "stomach", "belly", "gut", "arm", "leg", "shoulder", "knee", "hip",
    "back", "throat", "joint", "muscle", "artery", "vein",
    # Imaging Modalities & Technical Meta-words
    "xray", "ray", "ct", "mri", "scan", "ultrasound", "echo", "image",
    "film", "view", "axial", "sagittal", "coronal", "contrast",
    # Visual Descriptors (Common in VQA answering patterns)
    "white", "black", "grey", "gray", "dark", "bright", "clear",
    "blurry", "cloudy", "line", "lines", "circle", "round", "side",
    "left", "right", "top", "bottom", "upper", "lower", "middle",
    # Additional dataset-specific findings and anatomy
    "abcess", "abdomen", "abdominal", "abscess", "adenopathy", "adjacent", "air", "anterior", "aorta", "appendix", "ascending", "ascites", "asymmetric", "atherosclerotic", "basal", "base", "basilar", "biconvex", "biopsy", "blunting", "bowel", "breasts", "bronchiectasis", "bullous", "calcifications", "calcified", "cancer", "cardiac", "cardiopulmonary", "cardiovascular", "cartilage", "catheter", "caudate", "cavum", "cecum", "central", "cerebellum", "cerebrum", "choroid", "cirrhosis", "cns", "concave", "congenital", "cortical", "costophrenic", "crescent", "csf", "cva", "cxr", "cystic", "descending", "diffuse", "diffusion", "distal", "diverticuli", "diverticulitis", "dwi", "edematous", "elliptical", "embolus", "emphysema", "enhancement", "epidural", "exophytic", "extraluminal", "extremities", "fat", "fatty", "flair", "focal", "frontal", "gadolinium", "gallbladder", "gallstones", "gastrointestinal", "genetic", "haustra", "hemorrhage", "hepatocellular", "heterogeneous", "horsehoe", "hydrocephalus", "hydropneumothorax", "hyperintense", "hyperintensity", "hypodense", "hypointense", "hypoxic", "infarct", "infarcted", "infarcts", "infection", "infiltrative", "intestine", "irregular", "ischemia", "isointense", "ivc", "jejunum", "kidneys", "lateral", "lentiform", "location", "loculated", "maxillary", "medial", "mediport", "metastases", "metastasis", "micronodular", "multilobulated", "necrosis", "necrotic", "nephroblastomatosis", "nipple", "nodular", "nodules", "nucleus", "occipital", "oculomotor", "omental", "pacemaker", "pancreas", "pancreatic", "parasitic", "paratracheal", "parietal", "periappendiceal", "pericholecystic", "peritoneum", "pineal", "pituitary", "plicae", "pons", "portal", "psoas", "pulmonary", "punctate", "quadrantopia", "radiolucent", "respiratory", "retrocardiac", "retroperitoneum", "ruq", "sacroiliac", "scoliosis", "sella", "shrunken", "sigmoid", "sinusitis", "spleen", "splenule", "sternal", "sternotomy", "stones", "subarachnoid", "sulcal", "suprasellar", "temporal", "thalami", "thickening", "toxoplasma", "tumors", "ureteral", "varicocele", "vascular", "vasculature", "ventricle", "viral", "weighted"
]

# Minimum cumulative probability mass over medical seed tokens required
# for Gate 1 to pass.
# Tuning guide:
#   0.05 → very permissive (almost no false rejections)
#   0.10 → default         (good balance; rejects clearly non-medical text)
#   0.20 → strict          (may reject short or ambiguous clinical questions)
INTENT_THRESHOLD: float = 0.40

# ---------------------------------------------------------------------------
# Gate 2 – CLIP Medical Image Classifier
# ---------------------------------------------------------------------------

# Use the HuggingFace Hub directly instead of a local directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_MODEL_PATH: str = "openai/clip-vit-base-patch32"

# Minimum summed softmax score across all medical prompts for Gate 2 to pass.
# CLIP distributes probability across 10 medical + 10 non-medical prompts;
# a pure random image scores ~0.50, real medical images typically score >0.65.
# Tuning guide:
#   0.45 → permissive  (flags only obviously non-medical images)
#   0.55 → default     (good balance; rejects pets, nature photos, selfies)
#   0.70 → strict      (may reject low-quality or atypical scans)
CLIP_THRESHOLD: float = 0.60

# Kept for backward-compat with any code that imports these names.
CONFIDENCE_THRESHOLD: float = CLIP_THRESHOLD
TRIAGE_YES_THRESHOLD: float = CLIP_THRESHOLD

# ---------------------------------------------------------------------------
# VQA Model paths (mirrors inference.py — kept here so guardrail/ is
# self-contained and does not hard-code paths in multiple places)
# ---------------------------------------------------------------------------

CHECKPOINT_PATH: str = os.path.join(_REPO_ROOT, "med_pretrain_29_rad_34.pth")
CONFIG_PATH: str = os.path.join(_REPO_ROOT, "configs", "VQA.yaml")
TEXT_ENCODER: str = "bert-base-uncased"
TEXT_DECODER: str = "bert-base-uncased"
