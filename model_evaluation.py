import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def load_data(file_path):
    return pd.read_csv(file_path)

def load_model(model_path):
    return joblib.load(model_path)

def split_data(df):
    X = df.drop(columns=['Churn_Yes'])
    y = df['Churn_Yes']
    return X, y

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    file_path = 'processed_data.csv'
    model_path = 'churn_model.pkl'
    
    df = load_data(file_path)
    X, y = split_data(df)
    model = load_model(model_path)
    evaluate_model(model, X, y)
