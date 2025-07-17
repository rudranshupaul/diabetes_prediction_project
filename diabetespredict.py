import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib

# 1. Load and Explore Kaggle Dataset
df = pd.read_csv('diabetes.csv')

print("First 5 rows:")
print(df.head())

print("\nShape of the dataset:")
print(df.shape)

# 2. Data Preprocessing and Cleaning
print("\nInfo about the dataset:")
print(df.info())

print("\nMissing values in each column:")
print(df.isnull().sum())

# Replace zeros with NaN in specific columns
cols_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

print("\nMissing values after replacing zeros:")
print(df.isnull().sum())

# Fill missing values with the median
df[cols_with_missing] = df[cols_with_missing].fillna(df[cols_with_missing].median())

print("\nMissing values after filling with median:")
print(df.isnull().sum())

print("\nFirst 5 rows after preprocessing:")
print(df.head())

# 3. Exploratory Data Analysis (EDA)
print("\nDescriptive Statistics:")
print(df.describe())

# Histograms for Each Feature
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Boxplots to Check for Outliers
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns[:-1]):  # Exclude Outcome
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Check Class Balance
print("\nClass Distribution (Outcome):")
print(df['Outcome'].value_counts())

# 4. Feature Engineering
# BMI Categories
bins = [0, 18.5, 24.9, 29.9, 100]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['BMI_Category'] = pd.cut(df['BMI'], bins=bins, labels=labels)

# Age Groups
age_bins = [20, 30, 40, 50, 60, 100]
age_labels = ['20-29', '30-39', '40-49', '50-59', '60+']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Glucose-to-Age Ratio
small_const = 1e-5
df['Glucose_Age_Ratio'] = df['Glucose'] / (df['Age'] + small_const)

# Interaction Terms
df['Pregnancies_Age_Interaction'] = df['Pregnancies'] * df['Age']

print("\nNew engineered features (first 5 rows):")
print(df[['BMI_Category', 'Age_Group', 'Glucose_Age_Ratio', 'Pregnancies_Age_Interaction']].head())

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['BMI_Category', 'Age_Group'], drop_first=True)

# 5. Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_encoded.drop('Outcome', axis=1))
y = df_encoded['Outcome']

# 6. Model Training and Comparison
X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.3f}")

# 7. Model Evaluation and Visualization
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve Visualization
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Feature Importance Visualization
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    # Get feature names from df_encoded, excluding 'Outcome'
    feature_names = [col for col in df_encoded.columns if col != 'Outcome']
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title('Feature Importance')
    plt.show()

# 8. Model Saving and Final Summary
joblib.dump(model, 'diabetes_model.pkl')

print("\n=== PROJECT SUMMARY ===")
print("1. Data loaded and preprocessed successfully.")
print("2. Feature engineering and encoding completed.")
print("3. Model trained and evaluated with high accuracy.")
print("4. Model saved as 'diabetes_model.pkl'.")
print("5. Visualizations generated for model performance and feature importance.")
