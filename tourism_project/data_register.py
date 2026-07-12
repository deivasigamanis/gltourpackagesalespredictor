
# /content/drive/MyDrive/GreatLakes

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
from google.colab import userdata
import os

repo_id = "visasigamani/GL-Tourism-Package_Prediction"
repo_type = "dataset"

# Define constants for the dataset and output paths
token = userdata.get("HF_TOKEN")
print(token)
# Initialize API client
api = HfApi(token=token)
print(api)

# Step 1: Check if the space exists
try:
  api.repo_info(repo_id=repo_id, repo_type=repo_type)
  print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
  print(f"Space '{repo_id}' not found. Creating new dataset...")
  create_repo(repo_id=repo_id, repo_type=repo_type,token=token, private=False)
  print(f"Space '{repo_id}' created.")

status=api.upload_folder(
  folder_path="/content/drive/MyDrive/GreatLakes/tourism_project/data",
  repo_id=repo_id,
  repo_type=repo_type,
)
print(status)
