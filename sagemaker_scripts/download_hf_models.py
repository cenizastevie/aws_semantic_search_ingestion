import os
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer

MODELS = {
    "sentiment": {
        "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "model_class": AutoModelForSequenceClassification,
    },
    "summarization": {
        "model_id": "facebook/bart-large-cnn",
        "model_class": AutoModelForSeq2SeqLM,
    },
}

os.makedirs("models", exist_ok=True)

for name, info in MODELS.items():
    print(f"Downloading {name} model: {info['model_id']}")
    model_dir = os.path.join("models", name)
    os.makedirs(model_dir, exist_ok=True)
    model = info["model_class"].from_pretrained(info["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(info["model_id"])
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

print("✅ Sentiment and summarization models downloaded to ./models/")
