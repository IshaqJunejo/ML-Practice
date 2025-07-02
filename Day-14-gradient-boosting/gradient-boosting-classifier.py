import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('../Day-04/breast_cancer_dataset.csv')

#print(data.head())
#print(data.dtypes)

# Removing irrelevant columns and Labelling the target column
data = data.drop(columns=['id', 'Unnamed: 32'])
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Splitting into features and target
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

X = StandardScaler().fit_transform(X)

# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=3, random_state=42)

model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100} %')

print()
print('Classification Report:')
print(classification_report(y_test, y_pred))

print()
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))