from sagemaker import image_uris

image_uri = image_uris.retrieve(
    framework="huggingface",
    region="us-east-1",
    version="4.26.0",
    image_scope="inference",
    base_framework_version="pytorch1.13.1",
    instance_type="ml.m5.large"
)
# 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04
print(f"Hugging Face image URI: {image_uri}")