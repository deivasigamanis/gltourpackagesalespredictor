
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model

# replace with your repoid
model_path = hf_hub_download(repo_id="visasigamani/GL-Tourism-Package-Prediction", filename="best_tourism_package_model_v1.joblib")

model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Sales Prediction App")
st.write("""
This application predicts the  potential buyers based on its customer interaction data.
Please enter the customer data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=0, max_value=120,step=1)
CityTier = st.number_input("City Tier", min_value=1, max_value=3,step=1)
TypeofContact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
Occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
Designation= st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
DurationOfPitch = st.number_input("Duration of Pitch", min_value=0, max_value=None,step=1)
Gender= st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Person Visiting", min_value=0, max_value=None,step=1)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=None,step=1)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=0, max_value=5,step=1)
ProductPitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])
MaritalStatus = st.selectbox("Marital Status", ['Single', 'Divorced', 'Married', 'Unmarried'])
NumberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=None,step=1)
Passport_choice = st.selectbox("Passport", ["No", "Yes"])
# Map to numeric value
Passport = 1 if Passport_choice == "Yes" else 0

PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=5,step=1)
OwnCar_choice = st.selectbox("Own Car", ["No", "Yes"])
# Map to numeric value
OwnCar = 1 if OwnCar_choice == "Yes" else 0

NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=None,step=1)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=None,step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "DurationOfPitch": DurationOfPitch,
    "Gender": Gender,  
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,  
    "PreferredPropertyStar": PreferredPropertyStar, 
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "Designation": Designation
}])



if st.button("Predict Sale"):
    # Show the DataFrame itself
    st.write("📊 Input DataFrame:")
    st.write(input_data)
    prediction = model.predict(input_data)[0]
    result = "Package Sold" if prediction == 1 else "Package not picked by the customer"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
