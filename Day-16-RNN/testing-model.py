import numpy as np
from tensorflow.keras.models import load_model
import json

# Loading the model
model1 = load_model("first_rnn.keras")
model2 = load_model("first_gru.keras")

# Loading character mapping
char_to_idx = None
idx_to_char = None
with open('char_to_index.json') as f:
    char_to_idx = json.load(f)

with open('index_to_char.json') as f:
    idx_to_char = json.load(f)

# Function to encode and decode text
def encode_string(s):
    return [char_to_idx[c] for c in s]

def decode_char(i):
    return idx_to_char[str(i)]

# Temperature based sampling
def sample_with_temperature(logits, temperature=1.0):
    # Avoid division by zero
    logits = np.asarray(logits).astype('float64')
    logits = logits / temperature

    # Convert to probabilities
    exp_preds = np.exp(logits - np.max(logits))
    probs = exp_preds / np.sum(exp_preds)

    # Sample based on probabilities
    return np.random.choice(len(logits), p=probs)


# Function to generate text
def generate_text(model, seed, n_next_char):
    input_indices = encode_string(seed)
    input_indices = np.array(input_indices)[np.newaxis, :]

    result = list(seed)

    for i in range(n_next_char):
        # Predicting next character
        prediction = model.predict(input_indices, verbose=0)

        last_pred = prediction[0, -1]

        next_index = sample_with_temperature(last_pred, 1.5)
        next_char = decode_char(next_index)

        # Appending the result
        result.append(next_char)

        # Update input for next prediction
        input_indices = np.append(input_indices[0], next_index)[-input_indices.shape[1]:]
        input_indices = input_indices[np.newaxis, :]
    
    return ''.join(result)


print()
print()
starting_string = "What is it that you came here lookin"
n_char = 20

print(f"Seed String: \"{starting_string}\"")
print(f"Generating next \"{n_char}\" Characters")

print()
print("Result based on Recurrent Neural Network")
result1 = generate_text(model1, starting_string, n_char)
print(result1)

print()
print("Result based on Gated Recurrent Unit")
result2 = generate_text(model2, starting_string, n_char)
print(result2)