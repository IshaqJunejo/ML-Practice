import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Load the dataset
dataset = pd.read_csv('diabetes.csv')

# print(dataset.head())
# print(dataset.info())
# print(dataset.describe())
# print(dataset.isnull().sum())

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, dataset[column].mean())

# Split the dataset into features and target variable, and training-testing sets
X = dataset.drop(columns=['Outcome'])
y = dataset['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

# Create the KNN classifier
knn = KNeighborsClassifier(n_neighbors=27, metric='manhattan', p=2)
# Fit the classifier to the training data
knn.fit(X_train, y_train)
# Make predictions on the test data
y_pred = knn.predict(X_test)

# Print the model parameters
print("Co-efficients:", knn.get_params())

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))