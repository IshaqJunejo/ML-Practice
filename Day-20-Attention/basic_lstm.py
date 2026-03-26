import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # I don't really need it, as I only have a CPU

def preprocess_data(df):
    all_chars = sorted(list(set("".join(df["surname"].astype(str)))))

    char_to_int = {char: i + 1 for i, char in enumerate(all_chars)}
    char_to_int['<PAD>'] = 0

    nationalities = sorted(df['nationality'].unique())
    ntlty_to_int = {ntlty: i for i, ntlty in enumerate(nationalities)}

    return char_to_int, ntlty_to_int

class NameDataset(Dataset):
    def __init__(self, df, char_to_int, ntlty_to_int, max_len):
        self.names = df['surname'].values
        self.labels = [ntlty_to_int[n] for n in df['nationality'].values]
        self.weights = df['census_count'].values.astype(np.float32)
        self.char_to_int = char_to_int
        self.max_len = max_len

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = str(self.names[idx])

        seq = [self.char_to_int.get(c, 0) for c in name][:self.max_len]

        padded_seq = seq + [0] * (self.max_len - len(seq))

        return (torch.tensor(padded_seq, dtype=torch.long), 
        torch.tensor(self.labels[idx], dtype=torch.long))

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, bidrection):
        super(LSTMClassifier, self).__init__()
        self.bidrection = bidrection
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidrection)

        fc_input_dim = hidden_dim * 2 if bidrection else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)

        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))

        lstm_out, (h_n, c_n) = self.lstm(embedded)

        if self.bidrection:
            hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            hidden = h_n[-1, :, :]

        output = self.fc(self.dropout(hidden))

        return output

# LOAD THE DATA FROM THE CSV
df = pd.read_csv("./surname-nationality.csv")

# SPLIT INTO TRAIN, TEST
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['nationality']
)

# Hyperparameters
MAX_LEN = 25
EMBED_DIM = 128
HIDDEN_DIM = 128
LR = 0.001
EPOCHS = 50

char_map, nationality_map = preprocess_data(df)

train_dataset = NameDataset(train_df, char_map, nationality_map, MAX_LEN)
test_dataset = NameDataset(test_df, char_map, nationality_map, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = LSTMClassifier(len(char_map), EMBED_DIM, HIDDEN_DIM, len(nationality_map), True)

criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for names, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(names)

        raw_loss = criterion(outputs, labels)

        loss = raw_loss.mean()
        
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {total_loss/len(train_loader):.4f}")

# Testing
print("\nEvaluating Model")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for names, labels in test_loader:
        names, labels = names.to(device), labels.to(device)
        outputs = model(names)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
print(f"Accuracy on Testing data: {(100 * correct / total):.2f}%")

# torch.save(model.state_dict(), "basic_lstm.pth")
torch.save(model.state_dict(), "bi_lstm.pth")

metadata = {
    "char_map": char_map,
    "nationality_map": nationality_map,
    "max_len": MAX_LEN,
    "hidden_dim": HIDDEN_DIM,
    "embed_dim": EMBED_DIM
}

# with open("metadata_basic_lstm.json", "w") as f:
with open("metadata_bi_lstm.json", "w") as f:
    json.dump(metadata, f)
