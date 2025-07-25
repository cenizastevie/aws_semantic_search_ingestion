import boto3
import os
import json
import csv
import io
from typing import BinaryIO

s3 = boto3.client('s3')
firehose = boto3.client('firehose')
output_bucket = os.environ.get('OUTPUT_BUCKET', 'sea-news-articles')
input_bucket = os.environ.get('INPUT_BUCKET', 'sea-warc-input')
firehose_stream_name = os.environ.get('KINESIS_FIREHOSE_STREAM', '')
is_local = os.environ.get('IS_LOCAL', 'true').lower() == 'true'
print(f'Running in {"local" if is_local else "remote"} mode')
def upload_file(file_path: str, key: str):
    """Upload a local file to S3."""
    s3.upload_file(file_path, output_bucket, key)

def send_firehose_record(record_data: dict):
    """Send a record to the configured Kinesis Firehose stream in CSV format."""
    if not firehose_stream_name:
        print('KINESIS_FIREHOSE_STREAM not set, skipping Firehose record send.')
        return
    
    # Convert record to CSV format
    csv_buffer = io.StringIO()
    fieldnames = ['timestamp', 's3_key', 'url', 'title', 'language', 'domain', 'warc_file', 'scrape_date', 'content_length', 'bucket']
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    
    # Write only the data row (no header)
    writer.writerow(record_data)
    csv_line = csv_buffer.getvalue()
    
    try:
        response = firehose.put_record(
            DeliveryStreamName=firehose_stream_name,
            Record={
                'Data': csv_line.encode('utf-8')
            }
        )
        print(f"Sent CSV record to Firehose: {response['RecordId']}")
    except Exception as e:
        print(f"Error sending record to Firehose: {e}")

def upload_bytes(data: bytes, 
                 key: str, url: str, 
                 title: str = 'title', 
                 language: str = 'en', 
                 domain: str = 'domain',
                 warc_file: str = 'warc_file',
                 scrape_date: str = 'unknown', 
    ):
    """Send article data and metadata to Firehose for batched CSV upload to S3."""
    
    # Send structured data to Firehose (no direct S3 upload)
    firehose_record = {
        'timestamp': scrape_date,
        'url': url,
        'title': title,
        'language': language,
        'domain': domain,
        'warc_file': warc_file,
        'scrape_date': scrape_date,
        'content_length': len(data),
        'bucket': output_bucket
    }
    
    send_firehose_record(firehose_record)

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
