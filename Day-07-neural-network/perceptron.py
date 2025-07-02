import pandas as pd
from tensorflow.keras import models, layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data_train = pd.read_csv('mnist_train.csv')
data_test = pd.read_csv('mnist_test.csv')

#print(data_train.head())
#print(data_test.head())

X_train = data_train.drop(columns=['label']) / 255.0
y_train = data_train['label']

X_test = data_test.drop(columns=['label']) / 255.0
y_test = data_test['label']

#print(X_train.shape)

# Build the model
perceptron = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=((28 * 28),)),
    layers.Dense(16, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Configure the model
perceptron.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
perceptron.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = perceptron.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

perceptron.save('mnist_perceptron.h5')

# Plotting the training accuracy and validation accuracy
train_history = perceptron.history.history['accuracy']
val_history = perceptron.history.history['val_accuracy']

plt.plot(train_history, label='Training Accuracy')
plt.plot(val_history, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()