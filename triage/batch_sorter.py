"""
triage/batch_sorter.py
----------------------
Implements the VQA-driven batch triage sorting feature.
"""
from typing import List, Dict
import logging
import torch
import time
from PIL import Image

# Import existing inference utilities
from inference import load_model, pre_question

logger = logging.getLogger(__name__)


class BatchTriageService:
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self._model = None
        self._tokenizer = None
        self._transform = None
        
        # The baseline query to evaluate abnormality
        self.query = "Are there abnormalities in this image?"

    def _load(self):
        if self._model is not None:
            return
            
        logger.info("Loading MUMC VQA model for batch triage on %s...", self.device)
        self._model, self._tokenizer, self._transform = load_model(self.device)
        logger.info("Batch triage model loaded successfully.")

    def sort_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Takes a list of image paths, runs batched VQA inference to evaluate 
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
        image_tensors = []
        for p in image_paths:
            try:
                img = Image.open(p).convert('RGB')
                image_tensors.append(self._transform(img))
            except Exception as e:
                logger.error("Failed to load image %s: %s", p, e)
                # Fallback zero tensor if image fails to load
                image_tensors.append(torch.zeros(3, 256, 256))
                
        image_tensor_batch = torch.stack(image_tensors).to(self.device)
        
        # 2. Preprocess text query
        processed_q = pre_question(self.query, 50)
        queries = [processed_q] * len(image_paths)
        q_enc = self._tokenizer(
            queries,
            padding='longest',
            truncation=True,
            max_length=25,
            return_tensors='pt',
        ).to(self.device)
        
        # 3. Batched Inference to extract Softmax probs
        with torch.no_grad():
            image_embeds = self._model.visual_encoder(image_tensor_batch)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            
            question_output = self._model.text_encoder(
                q_enc.input_ids,
                attention_mask=q_enc.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            bos_ids = torch.tensor([[self._tokenizer.cls_token_id]] * len(image_paths), device=self.device)
            
            decoder_output = self._model.text_decoder(
                bos_ids,
                encoder_hidden_states=question_output.last_hidden_state,
                encoder_attention_mask=q_enc.attention_mask,
                return_dict=True,
            )
            
            logits = decoder_output.logits[:, 0, :]
            probs = torch.softmax(logits, dim=-1)
            
            yes_id = self._tokenizer.convert_tokens_to_ids("yes")
            no_id = self._tokenizer.convert_tokens_to_ids("no")
            
            results = []
            for i in range(len(image_paths)):
                p_yes = probs[i, yes_id].item()
                p_no = probs[i, no_id].item()
                
                # Normalize probability to 0-1 range considering only yes/no
                score = p_yes / (p_yes + p_no + 1e-9)
                
                # Threshold for binary display (though queue is sorted continuously)
                is_abnormal = score > 0.5
                
                results.append({
                    "image_path": image_paths[i],
                    "score": score,
                    "is_abnormal": is_abnormal
                })
        
        t1 = time.perf_counter()
        logger.info("Batch processed %d images in %.2f seconds.", len(image_paths), t1 - t0)
        
        # 4. Sort descending (highest abnormality score first)
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return sorted_results
