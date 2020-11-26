import torch.nn as nn
import torch.nn.functional as F


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
        return x


class LeNetStudent(nn.Module):
    def __init__(self):
        super().__init__()
        # kernel_size = 2 was too strong, so 4
        self.conv1 = nn.Conv2d(1, 3, kernel_size=4)
        # Model with batchnorm was too strong reaching 0.973 accuracy
        # so I switched it off
        # self.bn = nn.BatchNorm2d(6)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(1875, 10)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = self.flatten(x)
        x = self.fc2(x)
        return x
