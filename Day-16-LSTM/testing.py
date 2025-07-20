import numpy as np
from tensorflow.keras.models import load_model
import json

# Loading the model
model = load_model("prototype.keras")

# Loading character mapping
char_to_idx = None
idx_to_char = None
with open('char_to_index.json') as f:
    char_to_idx = json.load(f)

idx_to_char = {idx : char for char, idx in char_to_idx.items()}

# Function to encode and decode text
def encode_string(s):
    return [char_to_idx[c] for c in s]

def decode_char(i):
    return idx_to_char[i]

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

        next_index = sample_with_temperature(last_pred, 0.01)
        next_char = decode_char(next_index)

        # Appending the result
        result.append(next_char)

        # Update input for next prediction
        input_indices = np.append(input_indices[0], next_index)[-input_indices.shape[1]:]
        input_indices = input_indices[np.newaxis, :]
    
    return ''.join(result)


print()
print()
starting_string0 = "To be or not to be "
starting_string1 = "Would you proceed especially "
starting_string2 = "What is that you came here for "
n_char = 25

print(f"Seed String 1: \"{starting_string0}\"")
print(f"Seed String 2: \"{starting_string1}\"")
print(f"Seed String 3: \"{starting_string2}\"")
print(f"Generating next \"{n_char}\" Characters")

print()
print("Results based on Long Short Term Memory Model")
result0 = generate_text(model, starting_string0.lower(), n_char)
print("1, ", result0)
result1 = generate_text(model, starting_string1.lower(), n_char)
print("2, ", result1)
result2 = generate_text(model, starting_string2.lower(), n_char)
print("3, ", result2)
