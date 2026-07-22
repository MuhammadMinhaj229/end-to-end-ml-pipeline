import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import mlflow
import joblib
import os

def generate_data():
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    feature_names = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['churn'] = y
    return df

def train():
    df = generate_data()
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("churn_prediction")
    with mlflow.start_run():
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/churn_model.pkl")
        try:
            mlflow.xgboost.log_model(model, "model")
        except Exception as e:
            print("Could not log model to MLFlow:", e)
        print(f"Model trained with accuracy: {accuracy}")

if __name__ == "__main__":
    train()
