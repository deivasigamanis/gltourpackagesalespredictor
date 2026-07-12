

# for data manipulation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

from google.colab import userdata


mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.set_tracking_uri("https://composed-italicize-bunkhouse.ngrok-free.dev")
mlflow.set_experiment("Tourism-Package-Experiment-Prod")

# api = HfApi()

token = userdata.get("HF_TOKEN")

print(token)

api = HfApi(token=token)


tour_package_sell_df = pd.read_csv("hf://datasets/visasigamani/GL-Tourism-Package_Prediction/tourism.csv")
print("Dataset loaded successfully.")

tour_package_sell_df = tour_package_sell_df.drop(tour_package_sell_df.columns[0], axis=1)
# Drop the unique identifier
tour_package_sell_df.drop(columns=['CustomerID'], inplace=True)

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


# List of categorical features in the dataset (original names, but already encoded in Xtrain/Xtest)
categorical_features = [
    'TypeofContact', # The method by which the customer was contacted (Company Invited or Self Inquiry)
    'Occupation', # Customer's occupation (e.g., Salaried, Freelancer)
    'Gender', # Gender of the customer (Male, Female)
    'ProductPitched', # The type of product pitched to the customer.
    'MaritalStatus', # Marital status of the customer (Single, Married, Divorced)
    'Designation' # Customer's designation in their current organization
]

# Split into X (features) and y (target)
X = tour_package_sell_df.drop(columns=[target_col])
y = tour_package_sell_df[target_col]


# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)


# define the RF model
rf_model = RandomForestClassifier(
    class_weight="balanced",  # auto-adjusts weights
    n_estimators=200,
    random_state=42
)

# Define hyperparameter grid
param_grid = {
    'randomforestclassifier__n_estimators': [25, 50, 100, 200],       # number of trees
    'randomforestclassifier__max_depth': [None,5, 10, 20],       # depth of trees
    'randomforestclassifier__min_samples_split': [2, 5, 10, 20],       # min samples to split a node
    'randomforestclassifier__min_samples_leaf': [1, 2, 3, 4],         # min samples per leaf
    'randomforestclassifier__bootstrap': [True, False]             # use bootstrap samples
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, rf_model)


# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV( estimator=model_pipeline, # Use the full pipeline as the estimator
          param_grid=param_grid,
          cv=5,
          n_jobs=-1,
          verbose=2
          )
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face

    # please replace with your repoid
    repo_id = "visasigamani/GL-Tourism-Package-Prediction"

    repo_type = "model"
    #token=os.getenv("HF_TOKEN")


    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new model space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_package_model_v1.joblib",
        path_in_repo="best_tourism_package_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
