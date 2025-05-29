import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 1. Load dataset
df = pd.read_csv(r'D:\Healthcare App\heart.csv')

# 2. Check for NaNs
print("Missing values in dataset:")
print(df.isnull().sum())

# Optional: Handle NaNs (if any)
df = df.dropna()  # Or use df.fillna(method='ffill') / df.fillna(0), etc.

# 3. Features and target
X = df.drop('condition', axis=1)
y = df['condition']

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Feature Engineering - Polynomial Features (degree 2)
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 6. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# 7. Data Balancing using SMOTE on training data only
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\nOriginal training set size: {X_train_scaled.shape[0]}, Balanced training set size: {X_train_balanced.shape[0]}")

# 8. Logistic Regression with hyperparameter tuning
log_reg = LogisticRegression(max_iter=1000)
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_balanced, y_train_balanced)

print("\nBest hyperparameters:", grid_search.best_params_)

# 9. Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for ROC-AUC

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. ROC-AUC Score and Curve
roc_score = roc_auc_score(y_test, y_proba)
print("ROC AUC Score:", roc_score)

RocCurveDisplay.from_estimator(best_model, X_test_scaled, y_test)
plt.title("ROC Curve")
plt.grid()
plt.show()

# 11. Save model, scaler, and polynomial transformer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('poly.pkl', 'wb') as poly_file:
    pickle.dump(poly, poly_file)

print("Model, scaler, and polynomial feature transformer saved successfully!")
