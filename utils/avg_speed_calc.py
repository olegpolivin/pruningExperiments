import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from loaders import get_loaders

batch_size_train = 2048
batch_size_test = 2048


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


def go_through_data(net, data_loader, device):

    net.eval()
    with torch.no_grad():
        for (idx, (x, t)) in enumerate(data_loader):
            x = net.forward(x.to(device))
            t = t.to(device)
    return 1


device = "cuda"
train_loader, _ = get_loaders(batch_size_train, batch_size_test)
net = LeNet().to(device)
net.load_state_dict(torch.load("models/exp1.pth"))

t0 = time.time()
for i in range(20):
    go_through_data(net, train_loader, device)

total_time = time.time() - t0
print(total_time, total_time / (i + 1))
# 46.204389810562134 9.240877962112426
