from transformers import pipeline

model_name = "typeform/distilbert-base-uncased-mnli"
classifier = pipeline("zero-shot-classification", model=model_name)

queries = [
    "What are the side effects of aspirin?",
    "How do I bake a cake?",
    "I have a headache and a fever.",
    "Can you help me with my math homework?",
    "Show me the fracture in this xray."
]

candidate_labels = ["medical question", "non-medical question"]

for q in queries:
    res = classifier(q, candidate_labels)
    scores = dict(zip(res["labels"], res["scores"]))
    print(f"Q: {q}")
    print(f"Medical: {scores['medical question']:.4f}")
    print()
