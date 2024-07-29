import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(df):
    X = df.drop(columns=['Churn_Yes'])
    y = df['Churn_Yes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    file_path = 'processed_data.csv'
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    joblib.dump(model, 'churn_model.pkl')
    print("Model training complete. Model saved to 'churn_model.pkl'.")
