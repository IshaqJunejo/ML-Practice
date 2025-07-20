import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = KNeighborsClassifier()

# grid
parameters = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2, 3, 4]
}

grid = GridSearchCV(model, parameters, cv=StratifiedKFold(n_splits=10), scoring='accuracy')
grid.fit(X_train, y_train)

# Best parameters
print("Best Params:", grid.best_params_)

# Evaluate
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Test Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nF1 Score:")
print(f1_score(y_test, y_pred))