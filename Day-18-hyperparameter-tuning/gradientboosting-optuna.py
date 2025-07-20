import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import optuna

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

# optuna objective
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 300)
    max_depth = trial.suggest_int("max_depth", 3, 6)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.7, 1.0)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    score = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring="accuracy")
    return score.mean()

# optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best parameters
print("Best parameters: ", study.best_params)
best_model = GradientBoostingClassifier(**study.best_params)
best_model.fit(X_train, y_train)

# Evaluate
y_pred = best_model.predict(X_test)

print("Test Accuracy: ", accuracy_score(y_test, y_pred))

print("\nClassification Report: ")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

print("\nF1 Score: ", f1_score(y_test, y_pred))