import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Split the dataset into features and target variable
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Create a Random Forest Classifier
forest = RandomForestClassifier(n_estimators=100, random_state=42)

# cross-validation (5-fold)
scores = cross_val_score(forest, X, y, cv=5)  

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())