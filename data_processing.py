import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)
    return df

def encode_features(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

def scale_features(df):
    scaler = StandardScaler()
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])
    joblib.dump(scaler, 'scaler.pkl')
    return df

def preprocess_data(file_path):
    df = load_data(file_path)
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)
    return df, df.columns

if __name__ == "__main__":
    file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    df, columns = preprocess_data(file_path)
    df.to_csv('processed_data.csv', index=False)
    joblib.dump(columns, 'model_columns.pkl')
    print("Data processing complete. Processed data saved to 'processed_data.csv'.")
