FROM public.ecr.aws/deep-learning-containers/pytorch-inference:2.5-cpu-py311-ec2

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/inference.py /opt/ml/model/code/inference.py

ENV SAGEMAKER_PROGRAM=inference.py