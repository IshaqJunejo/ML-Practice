import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn import tree
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('diabetes.csv')

# replacing 0 with mean for specific columns
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, dataset[column].mean())

# Split the dataset into features and target variable, and training-testing sets
X = dataset.drop(columns=['Outcome'])
y = dataset['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Decision Tree
model = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print the model parameters
print("Co-efficients:", model.get_params())

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

# Visualizing the Decision Tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], rounded=True, fontsize=14, impurity=False, proportion=True)
plt.title("Decision Tree Visualization")
plt.show()