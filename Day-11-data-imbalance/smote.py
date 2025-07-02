import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Split the data into features and target
X = data.drop(columns=['Class'])
y = data['Class']

X_scaled = StandardScaler().fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

# Applying SMOTE
overSample = SMOTE(random_state=42)
X_resampled, y_resampled = overSample.fit_resample(X_train, y_train)

# Creating a new Logistic Regression Model
print("With SMOTE")
print("Creating our Model ... ")
model_2 = LogisticRegression(max_iter=1000, random_state=42)

print("Training our model ... ")
model_2.fit(X_resampled, y_resampled)
print("Training completed")

# Make predictions
y_pred_resampled = model_2.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_resampled))

print()
print("Classification Report:")
print(classification_report(y_test, y_pred_resampled))