import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
import os


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=100)
        self.fc4 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    accuraccy_list = []
    for epoch in range(epochs):
        total = 0
        correct = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1} \nIteration:{i + 1} \nLoss:{loss}')
            with torch.no_grad():
                total += labels.size(0)
                _, prediction = torch.max(outputs, 1)
                correct += (prediction == labels).sum().item()
        print(f'\nAccuracy of network in epoch {epoch + 1}: {100 * correct / total}')
    writer.flush()


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.15,), (0.3081,))]
    )

    train = MNIST("data", download=True, transform=transform, train=True)
    test = MNIST("data", download=True, transform=transform, train=False)

    train_loader = DataLoader(train, 100, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, 100, shuffle=False, num_workers=0)

    writer = SummaryWriter()

    model = Model()
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, test_loader, criterion, optimizer)
    writer.close()

    if not os.path.exists('generated_model'):
        os.mkdir('generated_model')

    # Saving the weights only of the model
    torch.save(model.state_dict(), 'generated_model/mnist_state_dict.pt')