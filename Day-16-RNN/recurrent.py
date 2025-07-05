import tensorflow as tf
from tensorflow.keras import models, layers
import json
import os

with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total Characters: {len(text)}")

chars = sorted(list(set(text)))

print(f"Total Unique Characters: {len(chars)}")

# mappings char to int, and vice versa
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Saving character mapping on disk
with open('char_to_index.json', 'w') as f:
    json.dump(char_to_idx, f)

with open('index_to_char.json', 'w') as f:
    json.dump(idx_to_char, f)


# Encoding entire text sequence as sequence of integers
encoded_text = [char_to_idx[ch] for ch in text]

# print(encoded_text)

# Splitting into Training, Testing, and Validation Dataset
train_size = int(0.7 * len(encoded_text))
val_size = int(0.15 * len(encoded_text))
test_size = len(encoded_text) - train_size - val_size

train_data = encoded_text[:train_size]
val_data = encoded_text[train_size:train_size + val_size]
test_data = encoded_text[train_size + val_size:]

# Model parameters
seq_len = 100
batch_size = 64
buffer_size = 2000

# Function to create datasets from giving array and parameters
def create_dataset(data, seq_len, batch_size, buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    sequences = dataset.batch(seq_len + 1, drop_remainder=True)
    
    def split_input_target(seq):
        return seq[:-1], seq[1:]

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    return dataset

train_dataset = create_dataset(train_data, seq_len, batch_size, buffer_size)
val_dataset = create_dataset(val_data, seq_len, batch_size, buffer_size)
test_dataset = create_dataset(test_data, seq_len, batch_size, buffer_size)

# making the RNN Model
vocab_size = len(chars)
embedding_dim = 256
rnn_units = 256

model = models.Sequential([
    layers.Input(batch_shape=(batch_size, None)),
    layers.Embedding(vocab_size, embedding_dim),
    # layers.SimpleRNN(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform'),
    layers.GRU(rnn_units, return_sequences=True),
    layers.Dense(vocab_size)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create a callback that saves the model's weights
checkpoint_path = "checkpoints/gru/cp.weights.h5"
checkpoint_dir = "checkpoints/gru"

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

if os.path.exists(checkpoint_path + '.index'):
    print("Restoring weights from checkpoint...")
    model.load_weights(checkpoint_path)


# Training the model
history = model.fit(train_dataset, epochs=20, validation_data=val_dataset, callbacks=[checkpoint_cb])

# Testing the model
test_loss = model.evaluate(test_dataset)
print("Test loss:", test_loss)

model.save('first_gru.keras')