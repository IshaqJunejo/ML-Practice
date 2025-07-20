import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Load the dataset
dataset = pd.read_csv('../Day-05-knn-random-forest/diabetes.csv')

# Replacing the zeros with mean values
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, dataset[column].mean())

# Split the dataset into features and target variable
X = dataset.drop(columns=['Outcome'])
y = dataset['Outcome']

# Train, Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = DecisionTreeClassifier(random_state=42)

# grid
parameters = {
    'max_depth': [None, 5, 10, 15, 20, 25],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': [None, 'sqrt', 'log2'],
    #'criterion': ['gini', 'entropy'] # The choice of 'entropy' is commented because it has been observed to be vulnerable to over-fitting
    'criterion': ['gini']
}

grid = GridSearchCV(model, parameters, cv=5, scoring='f1')
grid.fit(X_train, y_train)

# Best parameters
print("Best Parameters: ", grid.best_params_)

# Evaluate
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Test Accuracy: ", accuracy_score(y_test, y_pred))

print("\nClassification Report: ")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

print("\nF1 Score: ", f1_score(y_test, y_pred))