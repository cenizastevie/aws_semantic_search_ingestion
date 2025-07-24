import os
import json
import logging
import boto3
import torch
import csv
import io
import copy
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """
    Load the models for inference
    
    Args:
        model_dir (str): Directory where model artifacts are stored
    
    Returns:
        dict: Dictionary containing loaded models and clients
    """
    logger.info(f"Loading models from {model_dir}")
    
    # Get AWS region from environment variable or default to us-east-1
    aws_region = os.environ.get('AWS_REGION', 'us-east-1')
    
    # Load sentiment analysis model and tokenizer
    sentiment_path = os.path.join(model_dir, "sentiment")
    logger.info(f"Loading sentiment model from {sentiment_path}")
    
    if os.path.exists(sentiment_path):
        try:
            sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_path)
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_path)
            sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=sentiment_model, 
                tokenizer=sentiment_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            sentiment_pipeline = None
    else:
        logger.warning(f"Sentiment model path {sentiment_path} does not exist")
        sentiment_pipeline = None
    
    # Load summarization model and tokenizer
    summarization_path = os.path.join(model_dir, "summarization")
    logger.info(f"Loading summarization model from {summarization_path}")
    
    if os.path.exists(summarization_path):
        try:
            summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_path)
            summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_path)
            summarizer_pipeline = pipeline(
                "summarization", 
                model=summarization_model, 
                tokenizer=summarization_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading summarization model: {str(e)}")
            summarizer_pipeline = None
    else:
        logger.warning(f"Summarization model path {summarization_path} does not exist")
        summarizer_pipeline = None
    
    # Initialize Bedrock client for embeddings
    try:
        bedrock_runtime_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region
        )
        logger.info("Bedrock runtime client initialized")
    except Exception as e:
        logger.error(f"Error initializing Bedrock client: {str(e)}")
        bedrock_runtime_client = None
    
    # Return a dictionary of all loaded components
    return {
        "summarizer": summarizer_pipeline,
        "sentiment_analyzer": sentiment_pipeline,
        "bedrock_client": bedrock_runtime_client
    }

def input_fn(request_body, request_content_type):
    """
    Parse input data from CSV into a list of dicts.
    """
    logger.info(f"Received request with content type: {request_content_type}")
    if request_content_type == 'text/csv':
        try:
            csv_data = io.StringIO(request_body.decode('utf-8') if isinstance(request_body, bytes) else request_body)
            reader = csv.DictReader(csv_data)
            data = [row for row in reader]
            required_columns = ['url', 'title', 'language', 'domain', 'warc_file', 'scrape_date', 'content']
            for row in data:
                for col in required_columns:
                    if col not in row:
                        logger.error(f"Missing required column: {col}")
                        raise ValueError(f"Input CSV is missing required column: {col}")
            return data
        except Exception as e:
            logger.error(f"Error parsing CSV data: {str(e)}")
            raise ValueError(f"Error parsing CSV data: {str(e)}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}. Supported type is text/csv")

def predict_fn(input_data, model_dict):
    """
    Process the input data (list of dicts) and generate predictions
    """
    logger.info(f"Processing {len(input_data)} records")
    summarizer = model_dict.get("summarizer")
    sentiment_analyzer = model_dict.get("sentiment_analyzer")
    bedrock_client = model_dict.get("bedrock_client")
    output_data = copy.deepcopy(input_data)
    for idx, row in enumerate(output_data):
        try:
            content = row.get('content', '')
            row['summary'] = None
            row['sentiment'] = None
            row['sentiment_score'] = None
            row['embedding'] = None
            # Skip processing if content is empty
            if not isinstance(content, str) or not content.strip():
                logger.warning(f"Empty content for row {idx}")
                continue
            # 1. Generate summary
            summary = ""
            if summarizer:
                try:
                    max_length = 1024
                    truncated_content = content[:max_length] if len(content) > max_length else content
                    summary_result = summarizer(
                        truncated_content,
                        max_length=150,
                        min_length=30,
                        do_sample=False
                    )
                    summary = summary_result[0]["summary_text"] if summary_result else ""
                    row['summary'] = summary
                    logger.info(f"Generated summary for row {idx}")
                except Exception as e:
                    logger.error(f"Error generating summary for row {idx}: {str(e)}")
            # 2. Perform sentiment analysis on original content
            if sentiment_analyzer:
                try:
                    max_length = 512
                    truncated_content = content[:max_length] if len(content) > max_length else content
                    sentiment_result = sentiment_analyzer(truncated_content)
                    if sentiment_result:
                        row['sentiment'] = sentiment_result[0]["label"]
                        row['sentiment_score'] = sentiment_result[0]["score"]
                        logger.info(f"Generated sentiment for row {idx}")
                except Exception as e:
                    logger.error(f"Error generating sentiment for row {idx}: {str(e)}")
            # 3. Generate embeddings from the summary (not the original content)
            if bedrock_client and summary:
                try:
                    model_id = "amazon.titan-embed-text-v1"
                    request_body = json.dumps({
                        "inputText": summary
                    })
                    response = bedrock_client.invoke_model(
                        modelId=model_id,
                        contentType="application/json",
                        accept="application/json",
                        body=request_body
                    )
                    response_body = json.loads(response.get("body").read())
                    embedding = response_body.get("embedding")
                    row['embedding'] = embedding
                    logger.info(f"Generated embedding for row {idx}")
                except Exception as e:
                    logger.error(f"Error generating embedding for row {idx}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
    return output_data

def output_fn(prediction_data, response_content_type):
    """
    Format the prediction data (list of dicts) as output
    """
    logger.info(f"Formatting output with content type: {response_content_type}")
    if response_content_type == 'text/csv':
        try:
            if not prediction_data:
                return ""
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=prediction_data[0].keys())
            writer.writeheader()
            writer.writerows(prediction_data)
            return output.getvalue()
        except Exception as e:
            logger.error(f"Error converting results to CSV: {str(e)}")
            return "Error converting results to CSV"
    elif response_content_type == 'application/json':
        try:
            return json.dumps(prediction_data)
        except Exception as e:
            logger.error(f"Error converting results to JSON: {str(e)}")
            return json.dumps({"error": "Error converting results to JSON"})
    else:
        logger.warning(f"Unsupported content type: {response_content_type}, defaulting to text/csv")
        if not prediction_data:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=prediction_data[0].keys())
        writer.writeheader()
        writer.writerows(prediction_data)
        return output.getvalue()
