import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, lstm_out):
        attn_weights = self.attn(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        context, weights = self.attention(lstm_out)
        output = self.fc(self.dropout(context))
        return output

class LSTMBasic(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, bidrection):
        super(LSTMBasic, self).__init__()
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

def calculate_top_k(loader, model, device, k=3):
    model.eval()
    correct_1 = 0
    correct_k = 0
    total = 0

    with torch.no_grad():
        for names, labels in loader:
            names, labels = names.to(device), labels.to(device)
            outputs = model(names)
            
            # Top-1 Accuracy
            _, pred_1 = torch.max(outputs, 1)
            correct_1 += (pred_1 == labels).sum().item()

            # Top-K Accuracy
            _, pred_k = outputs.topk(k, 1, True, True)
            correct_k += torch.eq(pred_k, labels.view(-1, 1)).sum().item()

            total += labels.size(0)

    acc1 = (correct_1 / total) * 100
    acck = (correct_k / total) * 100
    return acc1, acck

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Metadata
    with open("metadata_basic_lstm.json", "r") as f:
        meta1 = json.load(f)
    with open("metadata_bi_lstm.json", "r") as f:
        meta2 = json.load(f)
    with open("metadata_attention_lstm.json", "r") as f:
        meta3 = json.load(f)
    
    # Create the Models
    model1 = LSTMBasic(
        len(meta1['char_map']), 
        meta1['embed_dim'], 
        meta1['hidden_dim'], 
        len(meta1['nationality_map']),
        False
    ).to(device)

    model2 = LSTMBasic(
        len(meta2['char_map']), 
        meta2['embed_dim'], 
        meta2['hidden_dim'], 
        len(meta2['nationality_map']),
        True
    ).to(device)

    model3 = LSTMClassifier(
        len(meta3['char_map']),
        meta3['embed_dim'],
        meta3['hidden_dim'],
        len(meta3['nationality_map'])
    ).to(device)

    # Load Weights for all models
    model1.load_state_dict(torch.load("basic_lstm.pth", map_location=device))
    model2.load_state_dict(torch.load("bi_lstm.pth", map_location=device))
    model3.load_state_dict(torch.load("attention_lstm.pth", map_location=device))

    df = pd.read_csv("./surname-nationality.csv")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['nationality']
    )

    test_dataset = NameDataset(test_df, meta1['char_map'], meta1['nationality_map'], meta1['max_len'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Calculate
    acc1, acc3 = calculate_top_k(test_loader, model1, device, k=3)

    print("\nBasic LSTM:")
    print("-" * 30)
    print(f"Top-1 Accuracy: {acc1:.2f}%")
    print(f"Top-3 Accuracy: {acc3:.2f}%")
    print("-" * 30)

    acc1, acc3 = calculate_top_k(test_loader, model2, device, k=3)

    print("\nBidirectional LSTM:")
    print("-" * 30)
    print(f"Top-1 Accuracy: {acc1:.2f}%")
    print(f"Top-3 Accuracy: {acc3:.2f}%")
    print("-" * 30)

    acc1, acc3 = calculate_top_k(test_loader, model3, device, k=3)

    print("\nAttention based LSTM:")
    print("-" * 30)
    print(f"Top-1 Accuracy: {acc1:.2f}%")
    print(f"Top-3 Accuracy: {acc3:.2f}%")
    print("-" * 30)