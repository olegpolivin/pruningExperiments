#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, 1)


def train(net, optimizer, data_loader, device):
    net.train()
    for (idx, (x, t)) in enumerate(data_loader):
        optimizer.zero_grad()
        x = net.forward(x.to(device))
        t = t.to(device)
        loss = F.nll_loss(x, t)
        loss.backward()
        optimizer.step()


def test(net, data_loader, device):
    top1 = 0  # TODO compute top1
    correct_samples = 0
    total_samples = 0
    net.eval()
    with torch.no_grad():
        for (idx, (x, t)) in enumerate(data_loader):
            x = net.forward(x.to(device))
            t = t.to(device)
            _, indices = torch.max(x, 1)
            correct_samples += torch.sum(indices == t)
            total_samples += t.shape[0]

    top1 = float(correct_samples) / total_samples
    return top1


if __name__ == "__main__":
    nb_epoch = 80
    batch_size_train = 1024
    batch_size_test = 5120
    device = "cuda"  # change to 'cpu' if needed

    best_model = None
    best_acc = 0

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "~/.cache/database/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "~/.cache/database/",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_test,
        shuffle=False,
    )
    net = LeNet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(nb_epoch):
        train(net, optimizer, train_loader, device)
        test_top1 = test(net, test_loader, device)
        print(f"Epoch {epoch}. Top1 {test_top1:.4f}")
        if test_top1 > best_acc:
            best_model = net
            best_acc = test_top1

    torch.save(best_model.state_dict(), "models/best_acc.pth")
