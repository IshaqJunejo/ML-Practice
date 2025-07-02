import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('mnist_test.csv')

x = dataset.iloc[:, 1:] # Features
y = dataset.iloc[:, 0] # Labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Output the results
print("Training Features:")
print(x_train)

print()
print("Testing Features:")
print(x_test)

print()
print("Traning Labels:")
print(y_train)

print()
print("Testing Labels:")
print(y_test)
