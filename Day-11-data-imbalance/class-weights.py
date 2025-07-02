import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Split the data into features and target
X = data.drop(columns=['Class'])
y = data['Class']

X_scaled = StandardScaler().fit_transform(X)

# Calculating Class Weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
#print(class_weights)
class_weights_dict = {i: w for i, w in zip(np.unique(y), class_weights)}
#print(class_weights_dict)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

# Creating a Logistic Regression Model
print("Using Class Weights ... ")
print("Creating our Model ... ")
model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weights_dict)

print("Fitting the model ... ")
model.fit(X_train, y_train)
print("Model training completed.")

# Testing our Model
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print()
print("Classification Report:")
print(classification_report(y_test, y_pred))