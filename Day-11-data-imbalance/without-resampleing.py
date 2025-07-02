import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('creditcard.csv')

#print(data.head())
#print(data.info())
#print(data.describe())
#print(data.isnull().sum())

# Split the data into features and target
X = data.drop(columns=['Class'])
y = data['Class']

X_scaled = StandardScaler().fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

# Create a Logistic regression model
print("Without Resampling the Data ... ")
print("Creating our Model ... ")
model = LogisticRegression(max_iter=1000, random_state=42)

print("Fitting the model ... ")
model.fit(X_train, y_train)
print("Model training completed.")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print()
print("Classification Report:")
print(classification_report(y_test, y_pred))