FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
