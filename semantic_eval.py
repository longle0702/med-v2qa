import json
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.models import GeminiModel

# ── Config ───────────────────────────────────────────────────────────────────
predictions_path = '/mnt/Data/Long-Data/MUMC_v3/output_v3/rad/result/med_pretrain_29_vqa_result_31_compare.json'
testset_path     = '/mnt/Data/Long-Data/MUMC_v3/data_RAD/testset.json'

model   = "gemini-2.5-flash"
api_key = "AIzaSyBi9q27n6oMA623kEqIVEZKpjeWy5G1Qu8"  

# ── Initialise Gemini model for deepeval ─────────────────────────────────────
gemini = GeminiModel(model=model, api_key=api_key)

# ── Load data ────────────────────────────────────────────────────────────────
with open(predictions_path, 'r') as f:
    predictions = json.load(f)   # {qid_str: [ground_truth, prediction]}

with open(testset_path, 'r') as f:
    testset = json.load(f)       # list of dicts

# Build a lookup: qid -> metadata
qid_to_meta = {str(item['qid']): item for item in testset}

# ── DeepEval GEval metric ─────────────────────────────────────────────────────
semantic_metric = GEval(
    name="SemanticCorrectness",
    criteria=(
        "Determine whether the 'actual output' conveys the same meaning as the "
        "'expected output'. Focus on semantic equivalence, not exact wording. "
        "Common acceptable variations include abbreviations (e.g. 'pa' == "
        "'posterior anterior'), partial matches that capture the key concept "
        "(e.g. 'hypodense' ≈ 'hypodense lesion'), or synonymous phrasings. "
        "Score 1 if the meanings match, 0 otherwise."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,
                       LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
    model=gemini,
)

# ── Build test cases ──────────────────────────────────────────────────────────
test_cases = []
qid_list   = []
type_list  = []

for qid_str, (ground_truth, prediction) in predictions.items():
    meta        = qid_to_meta.get(qid_str, {})
    answer_type = meta.get('answer_type', 'UNKNOWN').upper()
    question    = meta.get('question', '')

    tc = LLMTestCase(
        input=question,
        actual_output=str(prediction).strip().lower(),
        expected_output=str(ground_truth).strip().lower(),
    )
    test_cases.append(tc)
    qid_list.append(qid_str)
    type_list.append(answer_type)

# ── Run batch evaluation ──────────────────────────────────────────────────────
print(f"Evaluating {len(test_cases)} predictions with GEval (Gemini) ...\n")
eval_results = evaluate(
    test_cases=test_cases,
    metrics=[semantic_metric],
    print_results=False
)

# ── Process and Parse Metrics Results ─────────────────────────────────────────
results_correct = []
scores          = []

for res in eval_results:
    # Safely pull scores calculated from the evaluation batch
    metric_data = res.metrics_data[0] # Since we only passed one metric
    score = metric_data.score
    passed = score >= semantic_metric.threshold
    
    scores.append(score)
    results_correct.append(passed)

# ── Accuracy calculation ──────────────────────────────────────────────────────
def accuracy(flags):
    if not flags:
        return float('nan')
    return sum(flags) / len(flags) * 100

overall_flags = results_correct
open_flags    = [c for c, t in zip(results_correct, type_list) if t == 'OPEN']
closed_flags  = [c for c, t in zip(results_correct, type_list) if t == 'CLOSED']

overall_acc = accuracy(overall_flags)
open_acc    = accuracy(open_flags)
closed_acc  = accuracy(closed_flags)

print("\n" + "=" * 60)
print(f"  Semantic Evaluation Results (GEval · {model})")
print("=" * 60)
print(f"  Overall  : {sum(overall_flags):>4} / {len(overall_flags)}  →  {overall_acc:.2f}%")
print(f"  OPEN     : {sum(open_flags):>4} / {len(open_flags)}  →  {open_acc:.2f}%")
print(f"  CLOSED   : {sum(closed_flags):>4} / {len(closed_flags)}  →  {closed_acc:.2f}%")
print("=" * 60)

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    "model":        model,
    "overall_acc":  round(overall_acc, 2),
    "open_acc":     round(open_acc, 2),
    "closed_acc":   round(closed_acc, 2),
    "total":        len(overall_flags),
    "open_total":   len(open_flags),
    "closed_total": len(closed_flags),
    "per_sample": [
        {
            "qid":          qid,
            "answer_type":  atype,
            "ground_truth": tc.expected_output,
            "prediction":   tc.actual_output,
            "score":        round(s, 4),
            "correct":      correct,
        }
        for qid, atype, tc, s, correct in zip(
            qid_list, type_list, test_cases, scores, results_correct
        )
    ],
}

out_path = predictions_path.replace('_compare.json', '_semantic_eval.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n  Detailed results saved to:\n  {out_path}")