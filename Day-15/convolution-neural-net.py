import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

# Load the dataset
train_data = pd.read_csv('../Day-07/mnist_train.csv')
test_data = pd.read_csv('../Day-07/mnist_test.csv')

#print(train_data.head())

X_train = train_data.drop(columns=['label']).values
y_train = train_data['label'].values

X_test = test_data.drop(columns=['label']).values
y_test = test_data['label'].values

# normalize between [0, 1] and reshaping from (784, 1) to (28, 28, 1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Making the CNN Model
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test Accuracy: ", test_acc)

model.save('cnn-model.keras')