"""Train the Project 04-style Titanic decision tree and save model.joblib.

Run: python train_model.py
The script downloads Seaborn's Titanic example dataset on first use, so an internet
connection may be required. It uses the same feature order and sex encoding as app.py.
"""
from pathlib import Path

import joblib
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["pclass", "sex", "age", "fare"]
TARGET = "survived"
OUTPUT_PATH = Path(__file__).parent / "model.joblib"

def main():
    data = sns.load_dataset("titanic")[["survived", "pclass", "sex", "age", "fare"]].copy()
    data = data.dropna(subset=["survived", "pclass", "sex", "age", "fare"])
    data["sex"] = (data["sex"] == "female").astype(int)
    X = data[FEATURES]
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

    joblib.dump(model, OUTPUT_PATH)
    print(f"Saved model to: {OUTPUT_PATH}")
    print(f"Feature order saved for app: {FEATURES}")

if __name__ == "__main__":
    main()
