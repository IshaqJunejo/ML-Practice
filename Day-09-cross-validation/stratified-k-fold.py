import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv('../Day-04/breast_cancer_dataset.csv')
data = data.drop(columns=['Unnamed: 32', 'id'])

data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Split the dataset into features and target variable
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']
X = StandardScaler().fit_transform(X)

# Create models
logistic = LogisticRegression()
forest = RandomForestClassifier()
svm = SVC()
tree = DecisionTreeClassifier()
knn = KNeighborsClassifier()

# Create StratifiedKFold object
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform stratified k-fold cross-validation on Logistic Regression
scores_logistic = cross_val_score(logistic, X, y, cv=skf)

# Print the accuracy scores
print()
print("Logistic Regression:")
print(scores_logistic)
print("Mean Accuracy: ", scores_logistic.mean())

# Perform stratified k-fold cross-validation on Random Forest
score_forest = cross_val_score(forest, X, y, cv=skf)

# Print the accuracy scores
print()
print("Random Forest:")
print(score_forest)
print("Mean Accuracy: ", score_forest.mean())

# Perform stratified k-fold cross-validation on Support Vector Machine
score_svm = cross_val_score(svm, X, y, cv=skf)

# Print the accuracy scores
print()
print("Support Vector Machine:")
print(score_svm)
print("Mean Accuracy: ", score_svm.mean())

# Perform stratified k-fold cross-validation on Decision Tree
scores_tree = cross_val_score(tree, X, y, cv=skf)

# Print the accuracy scores
print()
print("Decision Tree:")
print(scores_tree)
print("Mean Accuracy: ", scores_tree.mean())

# Perform stratified k-fold cross-validation on K Nearest Neighbors
scores_knn = cross_val_score(knn, X, y, cv=skf)

# Print the accuracy scores
print()
print("K Nearest Neighbours:")
print(scores_knn)
print("Mean Accuracy: ", scores_knn.mean())

# Results:
# Logistic Regression: Mean Accuracy: 0.977 --- (Best Accuracy)
# Random Forest: Mean Accuracy: 0.959 --- (Most Time Consuming)
# Support Vector Machine: Mean Accuracy: 0.975
# Decision Tree: Mean Accuracy: 0.913 --- (Worst Accuracy)
# K Nearest Neighbours: Mean Accuracy: 0.970