import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
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

# Random Forest model
model = RandomForestClassifier(random_state=42)

# parameters
parameters = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced']
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameters,
    n_iter=100,
    cv=StratifiedKFold(n_splits=5),
    scoring='f1',
    n_jobs=-1,
    verbose=2,
    random_state=42

)
search.fit(X_train, y_train)

# Results
print("Best Parameters: ", search.best_params_)

# Evaluation
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("Test Accuracy: ", accuracy_score(y_test, y_pred))

print("\nClassification Report: ")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

print("\nF1 Score: ", f1_score(y_test, y_pred))