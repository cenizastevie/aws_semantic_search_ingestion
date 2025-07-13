import sys
from pathlib import Path
import pandas as pd
import boto3
import logging
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import predict_fn from the mounted inference.py for direct testing
sys.path.append(str(Path("..")))  # Add parent dir to sys.path for import
try:
    from inference import predict_fn
except ImportError as e:
    print(f"Could not import predict_fn from inference.py: {e}")

if __name__ == "__main__":
    # Load models
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("hf_models/sentiment", local_files_only=True)
    sentiment_tokenizer = AutoTokenizer.from_pretrained("hf_models/sentiment", local_files_only=True)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=sentiment_model,
        tokenizer=sentiment_tokenizer,
        device=-1
    )
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained("hf_models/summarization", local_files_only=True)
    summarization_tokenizer = AutoTokenizer.from_pretrained("hf_models/summarization", local_files_only=True)
    summarization_pipeline = pipeline(
        "summarization",
        model=summarization_model,
        tokenizer=summarization_tokenizer,
        device=-1
    )

    # Bedrock embedding client
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    bedrock_runtime_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=aws_region
    )
    logger.info("Bedrock runtime client initialized.")

    # Load CSV and prepare input
    df = pd.read_csv("test_input.csv")
    sample_row = df.iloc[0].to_dict()
    model_components = {
        "summarizer": summarization_pipeline,
        "sentiment_analyzer": sentiment_pipeline,
        "bedrock_client": bedrock_runtime_client
    }
    try:
        predict_results = predict_fn([sample_row], model_components)
        print("predict_fn output:")
        print(predict_results)
    except Exception as e:
        print(f"Error testing predict_fn: {e}")
