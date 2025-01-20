# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv(r"C:\Users\Mohammed Shoeb\spyder_10am_practice\logit classification.csv")

# Display basic information about the dataset (optional)
print("Dataset Head:\n", dataset.head())
print("\nDataset Info:\n")
dataset.info()

# Splitting the dataset into features and target variable
X = dataset.iloc[:, 2:4].values  # Features
y = dataset.iloc[:, -1].values   # Target variable

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictionss
y_pred = model.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

