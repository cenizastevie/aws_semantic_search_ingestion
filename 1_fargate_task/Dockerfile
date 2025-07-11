# Dockerfile for WARC File Extractor Batch Task
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies
RUN yum install -y gcc libxml2-devel libxslt-devel python3-devel

# Set working directory
WORKDIR /var/task

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Set the entrypoint for AWS Batch (adjust as needed)
CMD ["python", "main.py"]
