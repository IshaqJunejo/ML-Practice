import pandas as pd
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

#print(data.head())
#print(data.info())

# Convert the target variable to binary
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

# Perform k-fold cross-validation on Logistic Regression
scores_logistic = cross_val_score(logistic, X, y, cv=10)

# Print the accuracy scores
print()
print("Logistic Regression:")
print(scores_logistic)
print("Mean Accuracy: ", scores_logistic.mean())

# Perform k-fold cross-validation on Random Forest
scores_forest = cross_val_score(forest, X, y, cv=10)

# Print the accuracy scores
print()
print("Random Forest:")
print(scores_forest)
print("Mean Accuracy: ", scores_forest.mean())

# Perform k-fold cross-validation on SVM
scores_svm = cross_val_score(svm, X, y, cv=10)

# Print the accuracy scores
print()
print("Support Vector Machine:")
print(scores_svm)
print("Mean Accuracy: ", scores_svm.mean())

# Perform k-fold cross-validation on Decision Tree
scores_tree = cross_val_score(tree, X, y, cv=10)

# Print the accuracy scores
print()
print("Decision Tree:")
print(scores_tree)
print("Mean Accuracy: ", scores_tree.mean())

# Perform k-fold cross-validation on KNN
scores_knn = cross_val_score(knn, X, y, cv=10)

# Print the accuracy scores
print()
print("K-Nearest Neighbors:")
print(scores_knn)
print("Mean Accuracy: ", scores_knn.mean())

# Results:
# Logistic Regression: Mean Accuracy: 0.9806 --- (Best Accuracy)
# Random Forest: Mean Accuracy: 0.9578 --- (Most Time Consuming)
# Support Vector Machine: Mean Accuracy: 0.9770
# Decision Tree: Mean Accuracy: 0.9122 --- (Worst Accuracy)
# K-Nearest Neighbors: Mean Accuracy: 0.9666