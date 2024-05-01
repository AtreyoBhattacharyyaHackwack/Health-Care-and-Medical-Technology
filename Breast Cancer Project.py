import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the Breast Cancer Data Set
Breast_cancer_data = pd.read_csv('D:/python/Data/Breast_cancer_data.csv')

# Preprocess the data
# Handle missing Values
Breast_cancer_data['mean_radius'] = Breast_cancer_data['mean_radius'].fillna(Breast_cancer_data['mean_radius'].median())
Breast_cancer_data['mean_texture'] = Breast_cancer_data['mean_texture'].fillna(Breast_cancer_data['mean_texture'].median())

# Exploratory Data Analysis
print("Summary statistics:\n", Breast_cancer_data.describe())

# =============================================================================
# y = Breast_cancer_data['mean_radius']
# # Breast_cancer_data['diagnosis'] = Breast_cancer_data['diagnosis'].map({1: 'Malignant', 0: 'Benign'})
# 
# x = Breast_cancer_data['diagnosis']
# # print(Breast_cancer_data)
# =============================================================================

# Load the dataset

x = Breast_cancer_data[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']]
y = Breast_cancer_data['diagnosis']

# =============================================================================
# # Create bar chart
# plt.bar(x, y, color='skyblue')
# 
# # Adding labels and title
# plt.xlabel('Malignant/Benign')
# plt.ylabel('mean_radius')
# plt.title('Breast Cancer Analysis by radius of the cell')
# 
# # Show plot
# plt.show()
# =============================================================================

# =============================================================================
# # Create histogram
# plt.hist(y, bins=10, color='skyblue', edgecolor='black')
# 
# # Add labels and title
# plt.xlabel('mean_radius')
# plt.ylabel('No. of sample')
# plt.title('Breast Cancer Analysis by radius of the cell')
# 
# # Show plot
# plt.show()
# =============================================================================

# =============================================================================
# # Create box plot
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='diagnosis', y='mean_smoothness', data=Breast_cancer_data, palette='Set2')
# plt.title('Breast Cancer Analysis by mean_smoothness of the cell')
# plt.xlabel('Category')
# plt.ylabel('mean_smoothness')
# plt.show()
# =============================================================================

# =============================================================================
# # Create a scatter plot
# x = Breast_cancer_data['mean_texture']
# y = Breast_cancer_data['mean_smoothness']
# 
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=x, y=y, color='skyblue')
# plt.title('Breast Cancer Analysis: mean_texture vs mean_smoothness of the cell')
# plt.xlabel('mean_perimeter')
# plt.ylabel('mean_radius')
# plt.grid(True)
# plt.show()
# =============================================================================

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the data
scaler.fit(Breast_cancer_data)

# Transform the data
scaled_data = scaler.transform(Breast_cancer_data)

# =============================================================================
# print("Original Data:")
# print(Breast_cancer_data)
# 
# print("Scaled Data:")
# print(scaled_data)
# =============================================================================
# Calculate correlation matrix
correlation_matrix = Breast_cancer_data.corr()

# Plot correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Breast Cancer Analysis: Correlation Heatmap')
plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Selection
model = LogisticRegression()

# Training the model
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))



# Initialize k-fold cross-validation
# =============================================================================
# k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
# 
# # Initialize model
# model = LogisticRegression()
# 
# # Initialize list to store cross-validation scores
# cv_scores = []
# 
# # Perform k-fold cross-validation
# for train_idx, test_idx in k_fold.split(x):
#     # Split data into training and testing sets
#     X_train, X_test = x.iloc[train_idx], x.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#     
#     # Train model
#     model.fit(X_train, y_train)
#     
#     # Make predictions
#     y_pred = model.predict(X_test)
#     
#     # Calculate accuracy score
#     accuracy = accuracy_score(y_test, y_pred)
#     
#     # Append accuracy score to list
#     cv_scores.append(accuracy)
# 
# # Calculate mean and standard deviation of cross-validation scores
# mean_cv_score = np.mean(cv_scores)
# std_cv_score = np.std(cv_scores)
# 
# print("Cross-Validation Scores:", cv_scores)
# print("Mean CV Score:", mean_cv_score)
# print("Std CV Score:", std_cv_score)
# =============================================================================
