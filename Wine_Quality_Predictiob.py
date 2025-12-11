"""
Wine Quality Prediction ML Project
Author: Ashwini Fatkar
Objective: Predict wine quality using physicochemical properties.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Load Dataset 
# -------------------------------
df = pd.read_csv("winequality-red.csv")

print(df.head())

# -------------------------------
# Data Cleaning 
# -------------------------------
# Check nulls
print("Missing values:\n", df.isnull().sum())

df = df.dropna()

# -------------------------------
# Feature & Target
# -------------------------------
X = df.drop('quality', axis=1)
y = df['quality']

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model (Random Forest Regressor)
# -------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# -------------------------------
# Evaluate Model
# -------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nRandom Forest Regressor Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")

# -------------------------------
# Visualizations
# -------------------------------

## 1. Distribution of actual vs predicted quality
plt.figure(figsize=(8,6))
sns.histplot(y_test, color='blue', label='Actual', kde=True, stat="density", bins=10)
sns.histplot(y_pred, color='orange', label='Predicted', kde=True, stat="density", bins=10)
plt.title('Actual vs Predicted Wine Quality Distribution')
plt.xlabel('Quality Score')
plt.legend()
plt.show()

## 2. Feature Importance (top 5 features)
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(5)

plt.figure(figsize=(8,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Top 5 Features Influencing Wine Quality')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

## 3. Correlation Heatmap of features (including quality)
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap (Wine Dataset Features)')
plt.show()