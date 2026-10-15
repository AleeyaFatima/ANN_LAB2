"""
ANN Lab Task 02 - Task 1: Flood Prediction using Logistic Regression
Author: Aleeya Fatima (24108108)
Department of Robotics & Artificial Intelligence
SZABIST Islamabad
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    print("=" * 60)
    print("TASK 1: Flood Prediction using Logistic Regression")
    print("=" * 60)

    # Step 1: Read the dataset
    data = pd.read_csv("flood_dataset_classification.csv")
    print(f"Shape of dataset: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())

    # Step 2: Select features and target
    features = ["Rainfall", "Elevation", "Slope", "distance"]
    target = "occured"
    X = data[features]
    y = data[target]

    print("\nTarget distribution:")
    print(y.value_counts())

    # Step 3: Divide data into train (75%) and test (25%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=None
    )
    print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Step 4: Scale the values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Build and train the model
    clf = LogisticRegression()
    clf.fit(X_train_scaled, y_train)

    # Step 6: Evaluate the model
    predictions = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions) * 100
    cm = confusion_matrix(y_test, predictions)

    print("\n--- Model Evaluation ---")
    print(f"Model Intercept: {clf.intercept_[0]:.4f}")
    print("Learned Feature Coefficients:")
    for feat, coef in zip(features, clf.coef_[0]):
        print(f"  - {feat}: {coef:.4f}")

    print(f"\nAccuracy: {acc:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["No Flood (0)", "Flood (1)"]))

    # Step 7: Sample prediction test
    def flood_check(rainfall, elevation, slope, distance):
        sample = pd.DataFrame([[rainfall, elevation, slope, distance]], columns=features)
        sample_scaled = scaler.transform(sample)
        answer = clf.predict(sample_scaled)[0]
        prob = clf.predict_proba(sample_scaled)[0][1] * 100
        if answer == 1:
            return f"Flood Likely ({prob:.1f}% probability)"
        else:
            return f"No Flood Expected ({prob:.1f}% probability)"

    print("\n--- Sample Test Inferences ---")
    test_cases = [
        (120.0, 50.0, 2.5, 150.0),
        (25.0, 800.0, 15.0, 3500.0),
    ]
    for rf, el, sl, dist in test_cases:
        res = flood_check(rf, el, sl, dist)
        print(f"Inputs: Rainfall={rf}mm, Elevation={el}m, Slope={sl}, Distance={dist}m -> {res}")

if __name__ == "__main__":
    main()
