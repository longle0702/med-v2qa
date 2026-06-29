"""
triage/batch_sorter.py
----------------------
Implements the batch triage sorting feature using CLIP zero-shot classification.
"""
from typing import List, Dict
import logging
import torch
import time
from PIL import Image

from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
from guardrail.config import CLIP_MODEL_PATH

logger = logging.getLogger(__name__)

_ABNORMAL_PROMPTS = [
    "an abnormal medical scan",
    "a medical scan showing pathology",
    "an X-ray showing disease",
    "a clinical image with abnormalities",
]

_NORMAL_PROMPTS = [
    "a normal medical scan",
    "a healthy medical scan",
    "an X-ray with no abnormalities",
    "a healthy clinical image",
]


class BatchTriageService:
    def __init__(
        self,
        device: str = None,
        # Kept for compatibility but ignored
        preloaded_model=None,
        preloaded_tokenizer=None,
        preloaded_transform=None,
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self._clip_model = None
        self._clip_processor = None
        self.model_path = CLIP_MODEL_PATH

    def _load(self):
        if self._clip_model is not None:
            return
            
        logger.info("Loading CLIP model for batch triage on %s...", self.device)
        self._clip_model = CLIPModel.from_pretrained(self.model_path).to(self.device)
        self._clip_model.eval()
        self._clip_processor = CLIPProcessor.from_pretrained(self.model_path)
        logger.info("Batch triage CLIP model loaded successfully.")

    def sort_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Takes a list of image paths, runs batched CLIP inference to evaluate 
        abnormalities, and returns a sorted queue.
        
        Returns:
            List[Dict]: [{'image_path': str, 'score': float, 'is_abnormal': bool}, ...]
            Sorted by score in descending order (most abnormal first).
        """
        if not image_paths:
            return []
            
        self._load()
        
        t0 = time.perf_counter()
        
        # 1. Preprocess images
        images = []
        valid_paths = []
        for p in image_paths:
            try:
                img = Image.open(p).convert('RGB')
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                logger.error("Failed to load image %s: %s", p, e)
                # We skip failed images here or handle them with score 0
        
        if not images:
            return []

        all_prompts = _ABNORMAL_PROMPTS + _NORMAL_PROMPTS

        # 2. Batched Inference
        with torch.no_grad():
            inputs = self._clip_processor(
                text=all_prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self._clip_model(**inputs)
            
            # logits_per_image: (batch_size, num_prompts)
            logits = outputs.logits_per_image
            probs = F.softmax(logits, dim=1)
            
            n_abnormal = len(_ABNORMAL_PROMPTS)
            
            results = []
            for i, p in enumerate(valid_paths):
                # Sum probabilities for abnormal and normal classes
                abnormal_score = probs[i, :n_abnormal].sum().item()
                normal_score = probs[i, n_abnormal:].sum().item()
                
                # Normalize probability to 0-1 range
                score = abnormal_score / (abnormal_score + normal_score + 1e-9)
                
                # Threshold for binary display
                is_abnormal = score > 0.5
                
                results.append({
                    "image_path": p,
                    "score": score,
                    "is_abnormal": is_abnormal
                })
                
        # Add failed images with score 0
        failed_paths = set(image_paths) - set(valid_paths)
        for p in failed_paths:
            results.append({
                "image_path": p,
                "score": 0.0,
                "is_abnormal": False
            })
        
        t1 = time.perf_counter()
        logger.info("Batch processed %d images in %.2f seconds.", len(image_paths), t1 - t0)
        
        # 4. Sort descending (highest abnormality score first)
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return sorted_results
