import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

train_data = pd.read_csv("../Day-07-neural-network/mnist_train.csv")
test_data = pd.read_csv("../Day-07-neural-network/mnist_test.csv")

x_train = torch.tensor(train_data.iloc[:, 1:].values, dtype=torch.float32) / 255.0
y_train = torch.tensor(train_data.iloc[:, 0].values, dtype=torch.long)

x_test = torch.tensor(test_data.iloc[:, 1:].values, dtype=torch.float32) / 255.0
y_test = torch.tensor(test_data.iloc[:, 0].values, dtype=torch.long)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x


def train_net(model, train_loader, epochs):
    print("TRAINING")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(epochs):
        loss_display = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            # For display
            loss_display = loss.item()
        
        # if epoch % 20 == 0:
        print(f"Epoch {epoch+1}\nLoss: {loss:.4f}")

def test_net(model, test_loader):
    print("TESTING")
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, label in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    
    print(f"Accuracy: {(correct / len(test_loader.dataset) * 100):.4f}%")

# model = Net()

# train_net(model, train_loader, 5)
# test_net(model, test_loader)

# torch.save(model, "mnist_nn.pth")
# print("Model saved to disk")

model = torch.load("mnist_nn.pth", weights_only=False)

test_net(model, test_loader)