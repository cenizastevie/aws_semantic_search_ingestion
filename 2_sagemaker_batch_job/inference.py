import os
import json
import boto3
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models/clients (loaded once)
summarizer_pipeline = None # Using pipeline for simplicity
sentiment_pipeline = None # Using pipeline for simplicity
bedrock_runtime_client = None

def model_fn(model_dir):
    """
    Loads the models/clients for inference.
    `model_dir` is the path where model.tar.gz contents are extracted.
    """
    global summarizer_pipeline
    global sentiment_pipeline
    global bedrock_runtime_client

    logger.info(f"Loading models from: {model_dir}")

    # --- Load Summarization Model ---
    # Assuming your summarization model artifacts are in model_dir/summarization_model
    summarization_model_path = os.path.join(model_dir, "summarization_model")
    if os.path.exists(summarization_model_path):
        logger.info(f"Loading summarization model from {summarization_model_path}...")
        # Example: Load a Hugging Face summarization pipeline
        summarizer_pipeline = pipeline(
            "summarization",
            model=summarization_model_path,
            tokenizer=summarization_model_path,
            device=-1 # -1 for CPU, 0 for GPU if available and configured
        )
        logger.info("Summarization model loaded.")
    else:
        # Fallback or error if model not found
        logger.warning(f"Summarization model not found at {summarization_model_path}. "
                       "Summarization will not be performed or will rely on external API if implemented.")
        summarizer_pipeline = None # Or raise an error if summarization is mandatory

    # --- Load Sentiment Analysis Model ---
    # Assuming your sentiment model artifacts are in model_dir/sentiment_model
    sentiment_model_path = os.path.join(model_dir, "sentiment_model")
    if os.path.exists(sentiment_model_path):
        logger.info(f"Loading sentiment analysis model from {sentiment_model_path}...")
        # Example: Load a Hugging Face sentiment analysis pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=sentiment_model_path,
            tokenizer=sentiment_model_path,
            device=-1 # -1 for CPU, 0 for GPU if available and configured
        )
        logger.info("Sentiment analysis model loaded.")
    else:
        logger.warning(f"Sentiment analysis model not found at {sentiment_model_path}. "
                       "Sentiment analysis will not be performed or will rely on external API if implemented.")
        sentiment_pipeline = None # Or raise an error

    # --- Initialize Bedrock Client for Titan Embeddings ---
    # This does NOT load a model from model_dir; it sets up an API client.
    # The AWS_REGION environment variable is crucial here.
    aws_region = os.environ.get("AWS_REGION", "us-east-1") # Default to us-east-1 if not set
    logger.info(f"Initializing Bedrock runtime client for region: {aws_region}...")
    bedrock_runtime_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=aws_region
    )
    logger.info("Bedrock runtime client initialized.")

    # Return a dictionary of all loaded components
    return {
        "summarizer": summarizer_pipeline,
        "sentiment_analyzer": sentiment_pipeline,
        "bedrock_client": bedrock_runtime_client
    }

# ... (input_fn, predict_fn, output_fn remain largely the same, but use the global pipelines) ...

def predict_fn(input_data, model_components):
    """
    Accepts a list of dicts with keys: url, title, language, domain, warc_file, scrape_date, content.
    Performs summarization, sentiment analysis, and embedding generation on the 'summary' (not original content).
    Returns a list of dicts with keys: title, summary, sentiment_label, sentiment_score, embedding.
    """
    summarizer = model_components.get("summarizer")
    sentiment_analyzer = model_components.get("sentiment_analyzer")
    bedrock_client = model_components.get("bedrock_client")

    results = []
    for row in input_data:
        text = row.get("content", "")
        summary = ""
        sentiment_label = "UNKNOWN"
        sentiment_score = 0.0
        embedding = []

        # 1. Summarization
        if summarizer and text:
            try:
                summary_output = summarizer(text, max_length=150, min_length=30, do_sample=False)
                summary = summary_output[0]['summary_text'] if summary_output else ""
            except Exception as e:
                logger.error(f"Error during summarization: {e}")

        # 2. Sentiment Analysis (on summary)
        if sentiment_analyzer and summary:
            try:
                sentiment_output = sentiment_analyzer(summary)
                sentiment_label = sentiment_output[0]['label'] if sentiment_output else "UNKNOWN"
                sentiment_score = sentiment_output[0]['score'] if sentiment_output else 0.0
            except Exception as e:
                logger.error(f"Error during sentiment analysis: {e}")

        # 3. Titan Embeddings (on summary)
        if bedrock_client and summary:
            try:
                titan_embedding_payload = {
                    "inputText": summary
                }
                titan_embedding_response = bedrock_client.invoke_model(
                    body=json.dumps(titan_embedding_payload),
                    modelId="amazon.titan-embed-text-v1",
                    accept="application/json",
                    contentType="application/json"
                )
                embedding_body = json.loads(titan_embedding_response.get("body").read())
                embedding = embedding_body.get("embedding")
            except Exception as e:
                logger.error(f"Error generating Titan embedding: {e}")
                embedding = []
        else:
            if not bedrock_client:
                logger.warning("Bedrock client not initialized. Skipping Titan embeddings.")

        results.append({
            'title': row.get('title', ''),
            'summary': summary,
            'sentiment_label': sentiment_label,
            'sentiment_score': sentiment_score,
            'embedding': embedding
        })
    return results

def input_fn(request_body, request_content_type):
    """
    Parse input data for inference.
    For batch transform, this receives CSV data.
    """
    logger.info(f"Received content type: {request_content_type}")
    
    if request_content_type == 'text/csv':
        # Parse CSV input - expecting the format from your Firehose records
        import csv
        import io
        
        data = []
        csv_reader = csv.DictReader(io.StringIO(request_body))
        for row in csv_reader:
            data.append(row)
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def output_fn(prediction, content_type):
    """
    Format the prediction output.
    """
    logger.info(f"Formatting output with content type: {content_type}")
    
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")