# download_model.py
from huggingface_hub import snapshot_download
import os

# The model we want to use
model_name = "BAAI/bge-small-en-v1.5"

# The local directory where we want to save the model
local_dir = os.path.join("models", model_name.split("/")[-1])

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading model '{model_name}' to '{local_dir}'...")

# Download the model files
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False  # Important for Windows and deployment
)

print("Model download complete.")