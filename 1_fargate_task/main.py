import csv
import os
from urllib.parse import urlparse
from warcio.archiveiterator import ArchiveIterator
from newspaper import Article
from bs4 import BeautifulSoup
import re
import os
import boto3
from s3 import upload_bytes, get_input_file_stream, get_warc_file_stream
from langdetect import detect
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_filename(url):
    parsed = urlparse(url)
    path = parsed.path + (('_' + parsed.query) if parsed.query else '')
    if not path or path == '/':
        path = 'root'
    safe_path = re.sub(r'[^a-zA-Z0-9]', '_', path)
    if safe_path[0] == '_':
        safe_path = safe_path[1:]
    if safe_path.endswith('_'):
        safe_path = safe_path[:-1]
    return f'{safe_path}.txt'

def to_ascii(s):
    return s.encode('ascii', errors='ignore').decode('ascii')

def process_warc_stream(stream, warc_file):
    count = 0
    for idx, record in enumerate(ArchiveIterator(stream)):
        # if record.rec_type == 'metadata':
        #     scrape_date = record.rec_headers.get_header('WARC-Date')
        #     print(f"Scrape date for URL {url}: {scrape_date}")
        #     continue
        if record.rec_type != 'response':
            continue
        url = record.rec_headers.get_header('WARC-Target-URI')
        if not url:
            continue
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        if domain not in allowed_domains:
            continue
        payload = record.content_stream().read()
        scrape_date = record.rec_headers.get_header('WARC-Date')

        try:
            html = payload.decode('utf-8', errors='replace')
        except Exception as e:
            html = ''
            

        article = Article(url)
        article.set_html(html)
        try:
            article.parse()
            title = article.title
            article_text = article.text
        except Exception as e:
            title = ''
            article_text = ''
        if not article_text or len(article_text.strip()) < 500:
            soup = BeautifulSoup(html, 'html.parser')
            for script in soup(['script', 'style']):
                script.decompose()
            all_text = soup.get_text(separator=' ', strip=True)
            article_text = all_text
        # Detect language
        try:
            lang = detect(article_text)
        except Exception:
            lang = 'unknown'
        # if lang != 'en':
        #     print(f"Skipping non-English article (detected: {lang}) for URL: {url}")
        #     continue
        safe_domain = domain.replace('.', '_')
        filename = sanitize_filename(url)
        # Limit filename length to avoid S3 key too long error (e.g., 100 chars)
        max_filename_length = 100
        if len(filename) > max_filename_length:
            filename = filename[:max_filename_length]
        safe_title = to_ascii(title)
        safe_url = to_ascii(url)
        s3_key = f'{safe_domain}/{filename}'
        # Limit total key length (S3 max is 1024 bytes, but keep it much shorter)
        max_key_length = 200
        if len(s3_key) > max_key_length:
            s3_key = s3_key[:max_key_length]
        try:
            upload_bytes(
                f'{article_text}'.encode('utf-8'),
                s3_key,
                safe_url,
                safe_title,
                lang,
                safe_domain,
                os.path.basename(warc_file),
                scrape_date
            )
        except Exception as e:
            logger.error(f"Error uploading to S3 (key={s3_key}): {e}")

if __name__ == '__main__':
    batch_file_manifest = os.environ.get('BATCH_FILE_MANIFEST', 'batch_file_manifest_test.csv')
    batch_csv_stream = get_input_file_stream(batch_file_manifest)
    batch_csv_reader = csv.DictReader(batch_csv_stream.read().decode('utf-8').splitlines())
    warc_files = [row['wet_file_s3_path'].strip() for row in batch_csv_reader]

    allowed_domains = set()
    with open('domains.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            allowed_domains.add(row['domain'].strip())

    for warc_file in warc_files:
        logger.info(f'Processing WARC file: {warc_file}')
        with get_warc_file_stream(warc_file) as warc_stream:
            process_warc_stream(warc_stream, warc_file)
        logger.info(f'Finished processing WARC file: {warc_file}')
    logger.info("status completed")

# def handler(event, context):
#     batch_file_manifest = os.environ.get('BATCH_FILE_MANIFEST', 'batch_file_manifest_test.csv')
#     batch_csv_stream = get_input_file_stream(batch_file_manifest)
#     batch_csv_reader = csv.DictReader(batch_csv_stream.read().decode('utf-8').splitlines())
#     warc_files = [row['wet_file_s3_path'].strip() for row in batch_csv_reader]

#     allowed_domains = set()
#     with open('domains.csv', newline='', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             allowed_domains.add(row['domain'].strip())

#     for warc_file in warc_files:
#         print(f'Processing WARC file: {warc_file}')
#         with get_warc_file_stream(warc_file) as warc_stream:
#             process_warc_stream(warc_stream)
#         print(f'Finished processing WARC file: {warc_file}')
#     return {"status": "completed"}