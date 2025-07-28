import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Load dataset
dataset = pd.read_csv('../Day-02-data-processing/titanic_dataset.csv')

# Drop Irrelevant Columns
dataset.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], inplace=True, axis=1)

# Filling missing values
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode())

# Label Encoding "Categorical" values to "Numerical" values
labeler = LabelEncoder()
dataset['Embarked'] = labeler.fit_transform(dataset['Embarked'])
dataset['Sex'] = labeler.fit_transform(dataset['Sex'])

#
#print(dataset.head())

# Data splitting
X = dataset.drop(columns=['Survived'])
y = dataset['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
logistic = LogisticRegression()
tree = DecisionTreeClassifier()
forest = RandomForestClassifier()
svm = SVC()
neighbour = KNeighborsClassifier()

# Function
def predict_and_evaluate(model, test_x, test_y):
    y_pred = model.predict(test_x)

    print('Accuracy: ', accuracy_score(test_y, y_pred))
    print('\nClassification Report:')
    print(classification_report(test_y, y_pred))
    print('\nConfusion Matrix:')
    print(confusion_matrix(test_y, y_pred))
    print('\nF1 Score: ', f1_score(test_y, y_pred))

# Logistic Regression
print('Logistic Regression ...')
logistic.fit(X_train, y_train)

predict_and_evaluate(logistic, X_test, y_test)

# Decision Tree
print('\n\nDecision Tree ...')
tree.fit(X_train, y_train)

predict_and_evaluate(tree, X_test, y_test)

# Random Forest
print('\n\nRandom Forest ...')
forest.fit(X_train, y_train)

predict_and_evaluate(forest, X_test, y_test)

# Support Vector Machine
print('\n\nSVM ...')
svm.fit(X_train, y_train)

predict_and_evaluate(svm, X_test, y_test)

# KNN
print('\n\nK Nearest Neighbours ...')
neighbour.fit(X_train, y_train)

predict_and_evaluate(neighbour, X_test, y_test)