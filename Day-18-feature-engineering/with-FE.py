import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Load dataset
dataset = pd.read_csv('../Day-02-data-processing/titanic_dataset.csv')

## Feature Engineering
# Filling Null Values
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])

# Extracting Titles (from Names)
dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# print(dataset['Title'].value_counts())
dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Don', 'Lady', 'Sir', 'Capt', 'Countess', 'Jonkheer'], 'Rare')
dataset['Title'] = dataset['Title'].replace({'Ms' : 'Miss', 'Mlle' : 'Miss', 'Mme' : 'Miss'})

# Family Size
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

dataset['isAlone'] = 0
dataset.loc[dataset['FamilySize'] == 1, 'isAlone'] = 1

# Bining (used to help with outlier effect)
dataset['AgeBin'] = pd.cut(dataset['Age'], bins=[0, 12, 20, 40, 60, 80], labels=False)
dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels=False)

# Cabin Deck (First char of Cabin to represent Deck, 'U' for Unknow Cabins)
dataset['Deck'] = dataset['Cabin'].str[0]
dataset['Deck'] = dataset['Deck'].fillna('U')

# Label Encoding (for Gender)
dataset['Sex'] = dataset['Sex'].map({'male' : 0, "female" : 1})
# One-hot Encoding
dataset = pd.get_dummies(dataset, columns=['Embarked'], prefix='Embarked')
dataset = pd.get_dummies(dataset, columns=['Title'], prefix='Title')
dataset = pd.get_dummies(dataset, columns=['Deck'], prefix='Deck')


dataset.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'SibSp', 'Parch'], axis=1, inplace=True)
#print(dataset.dtypes)

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