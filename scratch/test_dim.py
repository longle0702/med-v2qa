import torch
from models.model_vqa import MUMC_VQA

config = {
    'distill': True,
    'image_res': 384,
    'bert_config': 'configs/config_bert.json'
}
model = MUMC_VQA(
    text_encoder='bert-base-uncased',
    text_decoder='bert-base-uncased',
    tokenizer=None,
    config=config
)

print("Model created successfully")
