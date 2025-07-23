
# Running SageMaker Inference Locally

## 1. Authenticate with AWS ECR
```sh
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
```

## 2. Pull the Inference Image
```sh
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04
```

## 3. Run with Docker Compose
```sh
docker compose up
```

> **Note:**
> If `docker compose up` does not work, check your Docker Compose file and ensure all required directories exist and are correctly mapped.