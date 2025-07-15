import os
import shutil
import tarfile

# Get directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths
hf_models_dir = os.path.join(current_dir, "hf_models")
output_tar_path = os.path.join(current_dir, "..", "model.tar.gz")
staging_dir = os.path.join(current_dir, "staging")

# Define the mapping: source folder → destination inside the tar
model_dirs = {
    "summarization": "summarization_model",
    "sentiment": "sentiment_model"
}

# Clean and create staging dir
if os.path.exists(staging_dir):
    shutil.rmtree(staging_dir)
os.makedirs(staging_dir)

# Copy each model into renamed folders in staging
for src, dest in model_dirs.items():
    src_path = os.path.join(hf_models_dir, src)
    dest_path = os.path.join(staging_dir, dest)
    shutil.copytree(src_path, dest_path)
    print(f"✓ Copied {src} to {dest}")

# Create model.tar.gz one level up from notebooks/
with tarfile.open(output_tar_path, "w:gz") as tar:
    tar.add(staging_dir, arcname="")  # Tar the contents directly
    print(f"📦 Created tarball at {output_tar_path}")

# Cleanup
shutil.rmtree(staging_dir)
print("🧹 Cleaned up temporary staging folder.")
