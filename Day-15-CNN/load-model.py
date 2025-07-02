import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the testing data
data = pd.read_csv('../Day-07/mnist_test.csv')

X_test = data.drop(columns=['label']).values / 255.0
y_test = data['label'].values

# Reshaping the Inputs
X_test = X_test.reshape(-1, 28, 28, 1)

# Load the model
model = load_model('cnn-model.keras')

# Predict using the model
y_pred = model.predict(X_test)
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

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_classes))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()