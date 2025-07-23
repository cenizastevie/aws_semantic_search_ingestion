import tarfile
import os

def create_model_tar_gz():
    # Create the tar.gz file
    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("models", arcname=".")
        tar.add("inference.py", arcname="code/inference.py")
    # Return the local path
    return os.path.abspath("model.tar.gz")

if __name__ == "__main__":
    print("Creating model package...")
    create_model_tar_gz()
    print("Model package created: model.tar.gz")