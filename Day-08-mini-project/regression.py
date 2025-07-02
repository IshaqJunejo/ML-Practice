import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('winequality-red.csv')

#print(data.head())
#print(data.info())
#print(data.describe())
#print(data.isnull().sum())

# Split the data into features and target variable
X = data.drop(columns=['quality'])
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a linear regression model
model = LinearRegression()

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print()

print("Coefficients:")
for i, col in enumerate(X.columns):
    print(f'{col}: {model.coef_[i]}')

# Plotting the co-efficients
plt.figure(figsize=(12, 6))
plt.barh(X.columns, model.coef_)
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients')
plt.axvline(0, color='red', linestyle='--')
plt.show()