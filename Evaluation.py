import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler  # for feature scaling
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Now here we are loading the dataset with the correct delimiter and check column names, and we are specifying the
# correct delimiter as ';' and load the dataset into a pandas DataFrame
data = pd.read_csv('winequality-red.csv', delimiter=';')

# Display column names to verify structure and avoid KeyErrors due to formatting
print("Column names:", data.columns)

# Now we can able to separate the features (input variables) and target (output variable) from the dataset
# Now we need to drop the target variable 'quality' to create a feature set and store 'quality' separately as the target
features = data.drop(columns=['quality'])  # separating input features
target = data['quality']  # extracting the target variable

# Now here we will verify whether features and target have been separated correctly or not.
print("Features:\n", features.head())  # print first few rows of features
print("Target:\n", target.head())  # print first few values of target variable


# Here we are categorizing the quality variable(Target Variable) into three classes:
# 0-3 -> bad, 4-6 -> average, 7-10 -> good
def classify_quality(value):
    if value <= 3:
        return 'Bad'
    elif value <= 6:
        return 'Average'
    else:
        return 'Good'


# Here we are applying function to create a new target variable
target = target.apply(classify_quality)


"""
In the Feature selection we are using three using types of methods which are Filter, Wrapper and Embedded.
The Filter method is used for to identify the correlated features, using correlation Matrix. 
The Wrapper method is used for to recursively select the important features using Support Vector Machine(SVM)
The Embedded method where we are using the Linear Kernel, to rank the feature importance using SVM. 


"""

# Now here we are checking the distribution of the target variable to identify whether any class is having any
# imbalance or not and then printing the counts of each unique value in "quality" to examine it, whether if some
# classes are insufficient.
print("Target variable distribution:\n", target.value_counts())

features_resampled, target_resampled = features, target

#  Now we can use the standardize features for optimal Support Vector machine performance
# after that we need to initialize the scaler for standardizing features
scaler = StandardScaler()  # create a StandardScaler object to standardize features

# Fit the scaler on the resampled feature data and transform it
features_resampled = scaler.fit_transform(
    features_resampled)  # These are the standard features to mean=0 and variance=1

# Verify the standardized data by printing the first few rows
print("Standardized features (first 5 rows):\n", features_resampled[:5])

# Filter Method - Correlation Matrix
# Here we are computing the correlation matrix to identify highly correlated features
correlation_matrix = pd.DataFrame(features_resampled).corr()
print("Correlation Matrix:\n", correlation_matrix)

# Here we are visualizing the highly correlated features if needed for removal
high_corr_threshold = 0.8
high_corr_features = [
    column for column in correlation_matrix.columns
    if any(correlation_matrix[column].abs() > high_corr_threshold) and column
]
print("Highly correlated features (above threshold of 0.8):", high_corr_features)

# We are using the Wrapper Method which is used for Recursive Feature Elimination with SVM
svc = SVC(kernel="linear")  # Use linear kernel for feature ranking
rfe = RFE(estimator=svc, n_features_to_select=5, step=1)  # Select top 5 features
rfe.fit(features_resampled, target_resampled)
selected_features_rfe = [feature for feature, selected in zip(data.columns[:-1], rfe.support_) if selected]
print("Top features selected by RFE:", selected_features_rfe)

# Embedded Method is used for the L1 Regularization with Linear SVM where we will use L1-penalized logistic
# regression, because it is used for to perform feature selection.
log_reg = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
log_reg.fit(features_resampled, target_resampled)

# Identify non-zero coefficients (features selected by L1 regularization)
selected_features_l1 = [
    feature for feature, coef in zip(data.columns[:-1], log_reg.coef_[0]) if coef != 0
]
print("Features selected by L1 regularization:", selected_features_l1)

"""
Now for the Part of  Support Vector Machine's implementation, 
1) We will split the data into training set and then testing set, then
2) We will set up an SVM model, after that
3) We will perform Hyperparameter Tunning, where we can use  kernel function like Linear Kernel
for non linear relationships to check the which kernel gives you the best accuracy.
4) In the end We will train the model and evaluate it on the basis of Classification Accuracy.
"""

# Use the selected features from Feature Selection
# Here, we'll use the features selected from RFE as an example
selected_features = selected_features_l1

# Filter the data to include only the selected features
X = pd.DataFrame(features_resampled, columns=data.columns[:-1])[selected_features]
y = target_resampled  # Target variable remains the same

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Kernel
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)  # Train model
y_pred_linear = svm_linear.predict(X_test)  # Make predictions
accuracy_linear = accuracy_score(y_test, y_pred_linear)  # Evaluate accuracy
print(f"Accuracy with linear kernel: {accuracy_linear:.2f}")

"""
Here we are implementing the K-fold cross validation to Evaluate SVM Kernel, which in return will give us a more robust
measure of performance by testing on multiple subset of data.
Here First we will set up the the K-Fold cross validation with classification accuracy as the metric.
Here we can use Scikit-Learn's StratifiedKFold Library to maintain the class distribution. After that we will be able to
apply the K-fold to evaluate accuracy across k fold for each kernel type.

"""

# Now we have to define the number of folds, here we are using the value as 5 because when need to go for low bias
# but higher variance, due to low data in each fold, which makes it to more sensitive to fluctuations. and 5 Fold
# Cross-validation is computationally less intensive than some higher values, due to which it will become useful when
# you are experimenting with multiple parameters.


k = 5

# Initialize Stratified K-Fold for maintaining class distribution in each fold
stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Linear Kernel
svm_linear = SVC(kernel='linear', random_state=42)
cv_scores_linear = cross_val_score(svm_linear, X, y, cv=stratified_kfold, scoring='accuracy')
print(f"Linear Kernel: Cross-Validation Accuracies: {cv_scores_linear}")
print(f"Linear Kernel: Average Accuracy over {k} folds: {cv_scores_linear.mean():.2f}")
