import tensorflow as tf
from tensorflow.keras import models, layers
import json
import os

with open("../Day-16-RNN/tiny-shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Convert the text to lowercase
text = text.lower()

chars = sorted(list(set(text)))

# character mapping
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Saving character mapping
with open('char_to_index.json', 'w') as f:
    json.dump(char_to_idx, f)

# Encoding text sequence
encoded_text = [char_to_idx[ch] for ch in text]

# Training, Testing, and Validation data
train_size = int(0.8 * len(encoded_text))
val_size = int(0.1 * len(encoded_text))
test_size = len(encoded_text) - train_size - val_size

train_data = encoded_text[:train_size]
val_data = encoded_text[train_size:train_size + val_size]
test_data = encoded_text[train_size + val_size:]

# Dataset parameters
seq_len = 128
batch_size = 64
buffer_size = 10000

# Function to create datasets
def create_dataset(data, seq_len, batch_size, buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    sequences = dataset.batch(seq_len + 1, drop_remainder=True)
    
    def split_input_target(seq):
        return seq[:-1], seq[1:]

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    return dataset

# Creating datasets
train_dataset = create_dataset(train_data, seq_len, batch_size, buffer_size)
val_dataset = create_dataset(val_data, seq_len, batch_size, buffer_size)
test_dataset = create_dataset(test_data, seq_len, batch_size, buffer_size)

# LSTM Model Parameters
vocab_size = len(chars)
embedding_dim = 128
units1 = 256
units2 = 246

# LSTM Model
model = models.Sequential([
    layers.Input(shape=(None, )),
    layers.Embedding(vocab_size, embedding_dim),
    layers.LSTM(units1, return_sequences=True),
    layers.LSTM(units2, return_sequences=True),
    layers.Dropout(0.2),
    layers.Dense(vocab_size, activation="softmax")
])

# Compile
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Checkpoint callback to save the model's weights
checkpoint_path = "checkpoints/cp.weights.h5"
checkpoint_dir = "checkpoints/"

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

if os.path.exists(checkpoint_path):
    print("Restoring weights from checkpoint...")
    model.load_weights(checkpoint_path)
else:
    print("Checkpoint not found")

# Learning Rate Scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-5
)

# Training the model
history = model.fit(
    train_dataset,
    epochs=100,
    initial_epoch=0,
    batch_size=batch_size,
    validation_data=val_dataset,
    callbacks=[checkpoint_cb, lr_scheduler]
)

# Testing the model
test_eval = model.evaluate(test_dataset)
print("Test Evaluation: ", test_eval)

model.save("lstm-prototype-3.0.keras")
