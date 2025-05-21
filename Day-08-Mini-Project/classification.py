import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('winequality-red.csv')
#print(data.head())

le = LabelEncoder()
data['quality'] = le.fit_transform(data['quality'])

# features and target variable
X = data.drop(columns=['quality'])
y = data['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
forest = RandomForestClassifier(n_estimators=150, random_state=42)

forest.fit(X_train, y_train)

Y_pred = forest.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, Y_pred), "\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, Y_pred), "\n")

print("Accuracy Score:", accuracy_score(y_test, Y_pred) * 100, " %\n")

# Plotting the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, Y_pred), display_labels=le.classes_)
disp.plot(cmap=plt.cm.Reds)
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
importances = forest.feature_importances_
indices = importances.argsort()[::-1]
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. feature {X.columns[indices[f]]} has importance {importances[indices[f]] * 100} %")

# Plot the feature importances 
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices])
plt.xlim([-1, X.shape[1]])
plt.show()