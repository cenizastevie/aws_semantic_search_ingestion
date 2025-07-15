# Dockerfile for WARC File Extractor ECS Fargate Task
FROM public.ecr.aws/docker/library/python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libxml2-dev \
    libxslt1-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Set the entrypoint for ECS Fargate
CMD ["python", "main.py"]
