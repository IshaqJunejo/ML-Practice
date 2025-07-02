import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
data_test = pd.read_csv('mnist_test.csv')

X_test = data_test.drop(columns=['label']) / 255.0
y_test = data_test['label']

# Load the model
perceptron = load_model('mnist_perceptron.h5')

# Evaluate the model
y_pred = perceptron.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Printing the Evaluation results
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_classes))

print()
print('Classification Report:')
print(classification_report(y_test, y_pred_classes))

print()
print('Accuracy:')
print(f'{(accuracy_score(y_test, y_pred_classes) * 100):.2f} %')