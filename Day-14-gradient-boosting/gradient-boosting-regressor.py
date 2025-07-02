import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error

# Load the dataset
data = pd.read_csv('../Day-08-Mini-Project/winequality-red.csv')

#print(data.head())
#print(data.dtypes)

# Splitting into features and target
X = data.drop(columns=['quality'])
y = data['quality']

X = StandardScaler().fit_transform(X)

# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2, max_depth=3, random_state=42)

model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)

print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')

y_pred_class = np.rint(y_pred).astype(int)

print()
print("After Rouding off for Classification")
print(f'Accuracy: {accuracy_score(y_test, y_pred_class) * 100} %')

print()
print("Classification Report:")
print(classification_report(y_test, y_pred_class))

print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))