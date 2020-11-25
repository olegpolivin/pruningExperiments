import torch
import torch.nn as nn
import torch.optim as optim

from student import LeNetStudent
from loaders import get_loaders


def train(net, loss_fn, optimizer, data_loader, device):
    net.train()
    for (idx, (x, t)) in enumerate(data_loader):
        optimizer.zero_grad()
        x = net.forward(x.to(device))
        t = t.to(device)
        loss = loss_fn(x, t)
        loss.backward()
        optimizer.step()


def test(net, data_loader, device):
    top1 = 0
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

    device = "cuda"

    net = LeNetStudent().to(device)

    batch_size_train = 512
    batch_size_test = 1024
    nb_epoch = 20

    train_loader, test_loader = get_loaders(batch_size_train, batch_size_test)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

    best_model = None
    best_acc = 0

    for epoch in range(nb_epoch):
        train(net, loss_fn, optimizer, train_loader, device)
        test_top1 = test(net, test_loader, device)
        print(f"Epoch {epoch}. Top1 {test_top1:.4f}")
        if test_top1 > best_acc:
            best_model = net
            best_acc = test_top1

    torch.save(best_model.state_dict(), "models/best_acc_student.pth")
