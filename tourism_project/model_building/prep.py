

import pandas as pd
import sklearn
import os

from huggingface_hub import HfApi
from google.colab import userdata

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import OneHotEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
token = userdata.get("HF_TOKEN")

api = HfApi(token=token)

# please create your dataset as you create your space
DATASET_PATH = "hf://datasets/visasigamani/GL-Tourism-Package_Prediction/tourism.csv"
print (DATASET_PATH)
df_tour = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unnamed column that contains row number 0-n
df_tour = df_tour.drop(df_tour.columns[0], axis=1)
# Drop the unique identifier
df_tour.drop(columns=['CustomerID'], inplace=True)

# Define the target variable for the task
target_col = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
     'Age',               # Customer's age
    'CityTier',            # City tier customer belogs
    'DurationOfPitch', # Duration of the sales pitch delivered to the customer.
    'NumberOfPersonVisiting',# Total number of people accompanying the customer on the trip.
    'NumberOfFollowups',     # Total number of follow-ups by the salesperson after the sales pitch
    'PreferredPropertyStar',         # Preferred hotel rating by the customer
    'NumberOfTrips',    # Average number of trips the customer takes annually
    'Passport' ,   # Whether the customer holds a valid passport (0: No, 1: Yes).
    'PitchSatisfactionScore', # Score indicating the customer's satisfaction with the sales pitch.
    'OwnCar', # Whether the customer owns a car (0: No, 1: Yes).
    'NumberOfChildrenVisiting', # :Number of children below age 5 accompanying the customer.
    'MonthlyIncome' # Gross monthly income of the customer
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact', # The method by which the customer was contacted (Company Invited or Self Inquiry)
    'Occupation', # Customer's occupation (e.g., Salaried, Freelancer)
    'Gender', # Gender of the customer (Male, Female)
    'ProductPitched', # The type of product pitched to the customer.
    'MaritalStatus', # Marital status of the customer (Single, Married, Divorced)
    'Designation' # Customer's designation in their current organization
]

# Encoding the categorical columns using onehot encoding
ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform all categorical features at once
encoded_features_array = ohe_encoder.fit_transform(df_tour[categorical_features])

# Create a DataFrame from the encoded features
encoded_features_df = pd.DataFrame(
    encoded_features_array,
    columns=ohe_encoder.get_feature_names_out(categorical_features),
    index=df_tour.index
)

# Drop the original categorical features from df_tour
df_tour = df_tour.drop(columns=categorical_features)

# Concatenate the original df_tour (now without original categorical features)
# with the new encoded DataFrame
df_tour = pd.concat([df_tour, encoded_features_df], axis=1)


# Define predictor matrix (X) using selected numeric and categorical features
# After one-hot encoding, X should include all numeric_features and the newly created encoded_features.
# So, X should be all columns except the target_col.
X = df_tour.drop(columns=[target_col])


# Define target variable
y = df_tour[target_col]


# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# repo id for dataset
dataset_repo_id="visasigamani/GL-Tourism-Package_Prediction"

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename

        repo_id=dataset_repo_id,

        repo_type="dataset",
    )


# List files in the dataset repository
files = api.list_repo_files(repo_id=dataset_repo_id, repo_type="dataset")

# Print the files uploaded  to validate and confirm
for f in files:
    print(f)
