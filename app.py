import streamlit as st
import pandas as pd
import joblib

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        st.write("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")

def preprocess_input(df):
    try:
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('model_columns.pkl')
        st.write("Scalers and columns loaded successfully")
        
        df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])
        df = pd.get_dummies(df, drop_first=True)
        
        # Add missing columns with default value of 0
        for col in columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[columns]
        return df
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")

def main():
    st.title("Customer Churn Prediction")
    
    tenure = st.number_input("Tenure", min_value=0)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    user_input = {
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method]
    }

    df = pd.DataFrame(user_input)
    st.write("User input collected")
    
    try:
        df = preprocess_input(df)
        st.write("Input preprocessed")
    except Exception as e:
        st.error(f"Error in preprocessing input: {e}")

    try:
        model = load_model('churn_model.pkl')
        if model:
            prediction = model.predict(df)
            prediction_prob = model.predict_proba(df)[:, 1]
        
            if prediction == 1:
                st.write("The customer is likely to churn.")
                st.write(f"Probability of churn: {prediction_prob[0]:.2f}")
            else:
                st.write("The customer is unlikely to churn.")
                st.write(f"Probability of churn: {prediction_prob[0]:.2f}")
    except Exception as e:
        st.error(f"Error in model prediction: {e}")

if __name__ == "__main__":
    main()
