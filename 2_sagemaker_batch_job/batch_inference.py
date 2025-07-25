import os
import json
import logging
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals for pipelines and client
summarizer_pipeline = None
sentiment_pipeline = None
bedrock_runtime_client = None

def model_fn(model_dir):
    """
    Load summarization and sentiment models, and initialize Bedrock client.
    """
    global summarizer_pipeline, sentiment_pipeline, bedrock_runtime_client

    # Summarization model
    summarization_model_path = os.path.join(model_dir, "summarization_model")
    if os.path.exists(summarization_model_path):
        summarizer_pipeline = pipeline(
            "summarization",
            model=summarization_model_path,
            tokenizer=summarization_model_path,
            device=-1
        )
        logger.info("Summarization model loaded.")
    else:
        summarizer_pipeline = None
        logger.warning(f"Summarization model not found at {summarization_model_path}.")

    # Sentiment model
    sentiment_model_path = os.path.join(model_dir, "sentiment_model")
    if os.path.exists(sentiment_model_path):
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=sentiment_model_path,
            tokenizer=sentiment_model_path,
            device=-1
        )
        logger.info("Sentiment analysis model loaded.")
    else:
        sentiment_pipeline = None
        logger.warning(f"Sentiment model not found at {sentiment_model_path}.")

    # Bedrock client for Titan embeddings
    import boto3
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    bedrock_runtime_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=aws_region
    )
    logger.info("Bedrock runtime client initialized.")

    return {
        "summarizer": summarizer_pipeline,
        "sentiment_analyzer": sentiment_pipeline,
        "bedrock_client": bedrock_runtime_client
    }

def input_fn(input_data, content_type):
    """
    Parse CSV input for batch transform.
    """
    logger.info(f"Received content type: {content_type}")
    if content_type == 'text/csv':
        import csv
        import io
        data = []
        csv_reader = csv.DictReader(io.StringIO(input_data))
        for row in csv_reader:
            data.append(row)
        return data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_components):
    """
    Run summarization, sentiment, and embedding on batch input.
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

        # Summarization
        if summarizer and text:
            try:
                summary_output = summarizer(text, max_length=150, min_length=30, do_sample=False)
                summary = summary_output[0]['summary_text'] if summary_output else ""
            except Exception as e:
                logger.error(f"Error during summarization: {e}")

        # Sentiment
        if sentiment_analyzer and summary:
            try:
                sentiment_output = sentiment_analyzer(summary)
                sentiment_label = sentiment_output[0]['label'] if sentiment_output else "UNKNOWN"
                sentiment_score = sentiment_output[0]['score'] if sentiment_output else 0.0
            except Exception as e:
                logger.error(f"Error during sentiment analysis: {e}")

        # Titan Embeddings
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

def output_fn(prediction, accept):
    """
    Format output as JSON.
    """
    logger.info(f"Formatting output with content type: {accept}")
    if accept == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {accept}")
