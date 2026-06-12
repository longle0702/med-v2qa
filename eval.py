import os
import sys
import json
import time
import datetime
from pathlib import Path

import torch
from ruamel.yaml import YAML

from models.model_vqa import MUMC_VQA
from models.vision.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
from vqaTools.vqa import VQA
from vqaTools.vqaEval import VQAEval

# ─────────────────────────── CONFIG ───────────────────────────
CHECKPOINT   = './output_v3/rad/med_pretrain_29_rad_31.pth'
DATASET_USE  = 'rad'
TEXT_ENCODER = 'bert-base-uncased'
TEXT_DECODER = 'bert-base-uncased'
DEVICE       = 'cuda'
SEED         = 42
CONFIG_PATH  = './configs/VQA.yaml'

# Output directory — saved under ./evaluation/<stem of checkpoint>/
_stem       = Path(CHECKPOINT).stem          # e.g. "med_pretrain_29_rad_35"
OUTPUT_DIR  = os.path.join('./evaluation', _stem)
RESULT_DIR  = os.path.join(OUTPUT_DIR, 'result')
# ──────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    """Run inference and collect per-question predictions."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        topk_ids, topk_probs = model(image, question, answer_list, train=False, k=config['k_test'])

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())
            _, pred = topk_prob.max(dim=0)
            result.append({"qid": ques_id, "answer": data_loader.dataset.answer_list[topk_id[pred]]})
    return result


def compute_accuracy(answer_list_path, result_file):
    """Evaluate a single result JSON with VQAEval and save accuracy metrics."""
    quesFile = answer_list_path

    try:
        vqa = VQA(quesFile, quesFile)
    except Exception as e:
        print(f"Failed to load annotation file: {quesFile}\nError: {e}")
        return None

    try:
        vqaRes  = vqa.loadRes(result_file, quesFile)
        vqaEval = VQAEval(vqa, vqaRes, n=2)
        vqaEval.evaluate()
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None

    acc = vqaEval.accuracy
    print(f"\nOverall Accuracy : {acc['overall']:.2f}")
    print("Per Answer Type  :")
    for ans_type, val in acc['perAnswerType'].items():
        print(f"  {ans_type} : {val:.2f}")

    # Save accuracy and per-question comparison
    acc_file     = result_file.replace('.json', '_acc.json')
    compare_file = result_file.replace('.json', '_compare.json')
    
    compare_data = {}
    for qid, answers in vqaEval.ansComp.items():
        gt, pred = answers
        qa_ann = vqa.qa.get(qid)
        ans_type = "unknown"
        if qa_ann and 'answer_type' in qa_ann:
            raw_type = str(qa_ann['answer_type']).lower()
            if raw_type in ('closed', 'close'):
                ans_type = 'close'
            elif raw_type == 'open':
                ans_type = 'open'
            else:
                ans_type = raw_type
        compare_data[qid] = {
            "type": ans_type,
            "gt": gt,
            "pred": pred
        }

    json.dump(acc,          open(acc_file,     'w'), indent=2)
    json.dump(compare_data, open(compare_file, 'w'), indent=2)
    print(f"\nAccuracy saved → {acc_file}")
    print(f"Comparison saved → {compare_file}")

    return acc


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    utils.set_seed(SEED)

    # ── Load config ──────────────────────────────────────────
    yaml_engine = YAML(typ='safe')
    with open(CONFIG_PATH, 'r') as f:
        config = yaml_engine.load(f)

    # ── Prepare output directories ───────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Result dir  : {RESULT_DIR}")

    # ── Redirect stdout → log file (mirrors notebook behaviour) ──
    sys.stdout = utils.Logger(
        filename=os.path.join(OUTPUT_DIR, 'eval_log.txt'),
        stream=sys.stdout
    )

    # ── Dataset ──────────────────────────────────────────────
    print(f"\nLoading dataset: {DATASET_USE}")
    datasets = create_dataset(DATASET_USE, config)
    print(f"  train size : {len(datasets[0])}")
    print(f"  test  size : {len(datasets[1])}")

    _, test_loader = create_loader(
        datasets,
        samplers=[None, None],
        batch_size=[config['batch_size_train'], config['batch_size_test']],
        num_workers=[4, 4],
        is_trains=[True, False],
        collate_fns=[vqa_collate_fn, None]
    )

    # ── Tokenizer & Model ────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(TEXT_ENCODER)
    print("\nBuilding model …")
    model = MUMC_VQA(
        config=config,
        text_encoder=TEXT_ENCODER,
        text_decoder=TEXT_DECODER,
        tokenizer=tokenizer
    )
    model = model.to(device)

    # ── Load checkpoint ──────────────────────────────────────
    print(f"Loading checkpoint: {CHECKPOINT}")
    checkpoint  = torch.load(CHECKPOINT, map_location='cpu')

    # The saved checkpoint may be a raw state-dict or wrapped in {'model': ...}
    state_dict  = checkpoint.get('model', checkpoint)

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Missing  keys : {msg.missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")

    # ── Inference ────────────────────────────────────────────
    print("\nRunning inference …")
    t0 = time.time()
    vqa_result = evaluation(model, test_loader, device, config)
    
    # Advanced temporal breakdown
    infer_time_seconds = time.time() - t0
    elapsed = str(datetime.timedelta(seconds=int(infer_time_seconds)))
    total_items = len(test_loader.dataset)
    avg_time_per_sample = infer_time_seconds / total_items if total_items > 0 else 0
    
    print(f"Inference time total : {elapsed}")
    print(f"Inference speed      : {avg_time_per_sample:.4f} seconds per sample")

    # ── Save raw predictions ─────────────────────────────────
    result_file = os.path.join(RESULT_DIR, f'{_stem}_vqa_result.json')
    json.dump(vqa_result, open(result_file, 'w'), indent=2)
    print(f"\nRaw predictions saved → {result_file}")

    # ── Evaluate ─────────────────────────────────────────────
    print("\nEvaluating Metrics …")
    answer_list_path = config[DATASET_USE]['test_file'][0]
    acc = compute_accuracy(answer_list_path, result_file)

    if acc is not None:
        summary = {
            'checkpoint': CHECKPOINT,
            'dataset': DATASET_USE,
            'overall_accuracy': acc['overall'],
            'per_answer_type': acc['perAnswerType'],
            'inference_metrics': {
                'total_time_formatted': elapsed,
                'total_time_seconds': infer_time_seconds,
                'avg_seconds_per_sample': avg_time_per_sample
            }
        }
        summary_file = os.path.join(OUTPUT_DIR, 'summary.json')
        json.dump(summary, open(summary_file, 'w'), indent=2)
        print(f"\nSummary saved → {summary_file}")
        
        # ── Updated Final Terminal View ───────────────────────
        print("\n===== EVALUATION SUMMARY =====")
        print(f"  Checkpoint   : {CHECKPOINT}")
        print(f"  Overall Acc  : {acc['overall']:.2f}")
        for k, v in acc['perAnswerType'].items():
            print(f"    {k:10s} : {v:.2f}")
        print(f"  Total Time   : {elapsed}")
        print(f"  Speed/Sample : {avg_time_per_sample:.4f}s")
        print("==============================")

if __name__ == '__main__':
    main()