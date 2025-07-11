import boto3
import os
from typing import BinaryIO

s3 = boto3.client('s3')
sqs = boto3.client('sqs')
output_bucket = os.environ.get('OUTPUT_BUCKET', 'sea-news-articles')
input_bucket = os.environ.get('INPUT_BUCKET', 'sea-warc-input')
sqs_queue_url = os.environ.get('SQS_QUEUE_URL', '')
is_local = os.environ.get('IS_LOCAL', 'true').lower() == 'true'
print(f'Running in {"local" if is_local else "remote"} mode')
def upload_file(file_path: str, key: str):
    """Upload a local file to S3."""
    s3.upload_file(file_path, output_bucket, key)

def send_sqs_message(message_body: str, message_attributes: dict = None):
    """Send a message to the configured SQS queue."""
    if not sqs_queue_url:
        print('SQS_QUEUE_URL not set, skipping SQS message send.')
        return
    sqs.send_message(
        QueueUrl=sqs_queue_url,
        MessageBody=message_body,
        MessageAttributes=message_attributes or {}
    )

def upload_bytes(data: bytes, 
                 key: str, url: str, 
                 title: str = 'title', 
                 language: str = 'en', 
                 domain: str = 'domain',
                 warc_file: str = 'warc_file',
                 scrape_date: str = 'unknown', 
    ):
    """Upload bytes data to S3 as an object and forward a message to SQS as notification."""
    s3.put_object(
        Bucket=output_bucket, 
        Key=key, 
        Body=data, 
        Metadata={
            'url': url, 
            'title': title, 
            'language': language, 
            'domain': domain,
            'warc_file': warc_file,
            'scrape_date': scrape_date
        }
    )
    
    message_body = f"Uploaded {key} from {url} (title: {title}, lang: {language})"
    send_sqs_message(message_body, {
        'Key': {'StringValue': key, 'DataType': 'String'},
        'URL': {'StringValue': url, 'DataType': 'String'},
        'Title': {'StringValue': title, 'DataType': 'String'},
        'Language': {'StringValue': language, 'DataType': 'String'},
        'Domain': {'StringValue': domain, 'DataType': 'String'},
        'WarcFile': {'StringValue': warc_file, 'DataType': 'String'},
        'ScrapeDate': {'StringValue': scrape_date, 'DataType': 'String'}
    })

def get_file_stream(bucket: str, key: str) -> BinaryIO:
    """Return a file-like stream for an S3 object."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj['Body']

def parse_s3_uri(s3_uri: str):
    """Parse an S3 URI into bucket and key."""
    if not s3_uri.startswith('s3://'):
        raise ValueError('Invalid S3 URI')
    parts = s3_uri[5:].split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    return bucket, key

def get_warc_file_stream(s3_uri: str) -> BinaryIO:
    """Return a stream for a WARC file in S3 using an S3 URI."""
    
    if is_local:
        return open('../../large_files/test.gz', 'rb')
    bucket, key = parse_s3_uri(s3_uri)
    return get_file_stream(bucket, key)

def get_input_file_stream(file_name: str) -> BinaryIO:
    """Get a stream for an input file in S3."""
    return get_file_stream(input_bucket, file_name)
