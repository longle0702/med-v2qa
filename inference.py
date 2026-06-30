# python inference.py --image test_inference_data/open/synpic18319.jpg --question "Describe the lung abnormalities?"

import argparse
import torch
from PIL import Image
from torchvision import transforms
from ruamel.yaml import YAML

from models.model_vqa import MUMC_VQA
from models.tokenization_bert import BertTokenizer
from dataset.utils import pre_question


CONFIG_PATH  = './configs/VQA.yaml'
CHECKPOINT   = './med_pretrain_29_rad_34.pth'
TEXT_ENCODER = 'bert-base-uncased'
TEXT_DECODER = 'bert-base-uncased'


def load_model(device):
    yaml_engine = YAML(typ='safe')
    with open(CONFIG_PATH, 'r') as f:
        config = yaml_engine.load(f)

    tokenizer = BertTokenizer.from_pretrained(TEXT_ENCODER)

    model = MUMC_VQA(
        config=config,
        text_encoder=TEXT_ENCODER,
        text_decoder=TEXT_DECODER,
        tokenizer=tokenizer,
    )
    model = model.to(device)

    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    transform = transforms.Compose([
        transforms.Resize(
            (config['image_res'], config['image_res']),
            interpolation=Image.BICUBIC,
        ),
        transforms.ToTensor(),
        normalize,
    ])

    return model, tokenizer, transform


def predict(model, tokenizer, transform, image_path, question_str,
            device, num_beams=3, max_new_tokens=20):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    processed_q = pre_question(question_str, 50)

    q_enc = tokenizer(
        [processed_q],
        padding='longest',
        truncation=True,
        max_length=25,
        return_tensors='pt',
    ).to(device)

    with torch.no_grad():
        image_embeds = model.visual_encoder(image_tensor)
        image_atts   = torch.ones(image_embeds.size()[:-1],
                                  dtype=torch.long).to(device)

        question_output = model.text_encoder(
            q_enc.input_ids,
            attention_mask=q_enc.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        bos_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)
        outputs = model.text_decoder.generate(
            input_ids=bos_ids,
            encoder_hidden_states=question_output.last_hidden_state,
            encoder_attention_mask=q_enc.attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(outputs[0][1:], skip_special_tokens=True).strip()
    return answer


def main():
    parser = argparse.ArgumentParser(
        description='Real-time VQA: give an image and a question, get an answer.'
    )
    parser.add_argument('--image',    required=True,  help='Path to the input image.')
    parser.add_argument('--question', required=True,  help='Question to ask about the image.')
    parser.add_argument('--num_beams',     type=int, default=3,
                        help='Beam width (1 = greedy, default: 3).')
    parser.add_argument('--max_new_tokens', type=int, default=20,
                        help='Max answer tokens (default: 20).')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'Loading model on {device}...')
    model, tokenizer, transform = load_model(device)

    answer = predict(
        model, tokenizer, transform,
        image_path=args.image,
        question_str=args.question,
        device=device,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    print(f'Question : {args.question}')
    print(f'Answer   : {answer}')


if __name__ == '__main__':
    main()
