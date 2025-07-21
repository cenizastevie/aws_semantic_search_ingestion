FROM public.ecr.aws/deep-learning-containers/pytorch-inference:2.5-cpu-py311-ec2

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SageMaker inference toolkit
RUN pip install sagemaker-inference

# Copy inference script to the correct location
COPY inference.py /opt/ml/code/inference.py

# Set environment variables
ENV SAGEMAKER_PROGRAM=inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code