import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
dataframe = pd.read_csv('breast_cancer_dataset.csv')

# print(dataframe.head())
# print(dataframe.info())
# print(dataframe.describe())
# print(dataframe.isnull().sum())

dataframe.drop(columns=['Unnamed: 32', 'id'], inplace=True)  # Drop the extra columns
# print(dataframe.isnull().sum())

dataframe['diagnosis'] = dataframe['diagnosis'].map({'M': 1, 'B': 0}) 

# print(dataframe['diagnosis'].value_counts())

# Split the dataset into features and target variable
X = dataframe['diagnosis']
y = dataframe.drop(columns=['diagnosis'])

# print(X.shape)
# print(y.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(y, X, test_size=0.2, random_state=42)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# Scaling the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(x_train_scaled, y_train)

# Print the model parameters
print('Model Coefficients:')
print(model.coef_)
print('Model Intercept:')
print(model.intercept_)

# Make predictions
y_pred = model.predict(x_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))