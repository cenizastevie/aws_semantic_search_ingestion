# Dockerfile
# Use a base image that includes Python and necessary ML frameworks
# Example for PyTorch and Hugging Face:
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SageMaker Inference Toolkit (essential for SageMaker container protocol)
# This provides the sagemaker-inference-toolkit entrypoint
RUN pip install sagemaker-inference-toolkit

# Copy your inference script
COPY inference.py /opt/ml/model/code/inference.py

# Set the entrypoint for SageMaker Inference Toolkit
# This will automatically call your inference.py functions
ENV SAGEMAKER_PROGRAM inference.py