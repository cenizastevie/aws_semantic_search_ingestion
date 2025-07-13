from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
import os

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

EMBEDDING_MODELS = {
    "embedding": {
        "model_id": "all-MiniLM-L6-v2",  # Lightweight, 384 dimensions
    }
}
os.makedirs("hf_models", exist_ok=True)

for name, info in MODELS.items():
    print(f"Downloading {name} model: {info['model_id']}")
    model_dir = f"hf_models/{name}"
    os.makedirs(model_dir, exist_ok=True)

    model = info["model_class"].from_pretrained(info["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(info["model_id"])

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

print("✅ Sentiment and summarization models downloaded to ./hf_models/")

for name, info in EMBEDDING_MODELS.items():
    print(f"Downloading {name} model: {info['model_id']}")
    model_dir = f"hf_models/{name}"
    os.makedirs(model_dir, exist_ok=True)
    
    # SentenceTransformer handles download and save differently
    embedding_model = SentenceTransformer(info["model_id"])
    embedding_model.save(model_dir)

print("✅ Sentiment, summarization, and embedding models downloaded to ./hf_models/")