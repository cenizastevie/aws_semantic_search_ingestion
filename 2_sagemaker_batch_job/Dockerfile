FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.13.1-cpu-py39-ubuntu20.04-sagemaker

RUN pip install pandas scikit-learn matplotlib

COPY batch_inference.py /opt/ml/code/
COPY model_handler.py /opt/ml/code/

WORKDIR /opt/ml/code

ENV SAGEMAKER_PROGRAM batch_inference.py
