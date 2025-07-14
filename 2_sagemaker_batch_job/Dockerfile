FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:pytorch-2.4.0-gpu-py311

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/inference.py /opt/ml/model/code/inference.py

ENV SAGEMAKER_PROGRAM=inference.py