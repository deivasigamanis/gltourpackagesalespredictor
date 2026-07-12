from huggingface_hub import HfApi
import os

from google.colab import userdata

token=userdata.get("HF_TOKEN")   # please use  token
print(token)
api = HfApi(token=token)

repo_id = "visasigamani/GL-Tourism-Package_Prediction"
folder_path = "/content/drive/MyDrive/GreatLakes/tourism_project/deployment"
repo_type = "space"


api.upload_folder(
    folder_path=folder_path,     # the local folder containing  files
    repo_id=repo_id,          # the target repo
    repo_type=repo_type,                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo

)
