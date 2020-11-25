import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import flops, model_size
from loaders import get_loaders
from student import LeNet, LeNetStudent


def train(teacher, student, loss_fn, optimizer, data_loader, device):
    teacher.train(False)
    student.train()
    for (idx, (x, t)) in enumerate(data_loader):
        optimizer.zero_grad()
        x = x.to(device)
        x_teacher = teacher.forward(x)
        x_student = student.forward(x)
        t = t.to(device)
        loss = loss_fn(x_student, x_teacher, t)
        loss.backward()
        optimizer.step()


def test(net, data_loader, device):
    top1 = 0
    correct_samples = 0
    total_samples = 0
    net.train(False)
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


def calculate_prune_metrics(net, test_loader, device):

    x, _ = next(iter(test_loader))
    x = x.to(device)

    size, size_nz = model_size(net)

    FLOPS = flops(net, x)
    return FLOPS, size, size_nz


def cross_entropy_with_soft_targets(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))


class CrossEntropyLossTemperature_withSoftTargets(torch.nn.Module):
    def __init__(self, temperature, reduction="mean"):
        super(CrossEntropyLossTemperature_withSoftTargets, self).__init__()
        self.T = temperature
        self.reduction = reduction

    def forward(self, input, soft_targets, hard_targets):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.
        """
        z = input / self.T
        loss_1 = self.T ** 2 * cross_entropy_with_soft_targets(
            z, F.softmax(soft_targets, 1, _stacklevel=5)
        )

        loss_2 = F.cross_entropy(
            input,
            hard_targets,
            weight=None,
            ignore_index=-100,
            reduction=self.reduction,
        )

        return loss_1 + loss_2


if __name__ == "__main__":

    device = "cuda"
    teacher = LeNet().to(device)
    student = LeNetStudent().to(device)

    teacher.load_state_dict(torch.load("models/best_acc.pth"))
    student.load_state_dict(torch.load("models/best_acc_student.pth"))

    batch_size_train = 512
    batch_size_test = 1024
    nb_epoch = 20

    train_loader, test_loader = get_loaders(batch_size_train, batch_size_test)

    print(calculate_prune_metrics(teacher, test_loader, device))
    print(calculate_prune_metrics(student, test_loader, device))

    loss_fn = CrossEntropyLossTemperature_withSoftTargets(1)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.0001, weight_decay=0.00001)

    best_model = None
    best_acc = 0

    teacher.train(False)
    for epoch in range(nb_epoch):
        train(teacher, student, loss_fn, optimizer, train_loader, device)
        test_top1 = test(student, test_loader, device)
        print(f"Epoch {epoch}. Top1 {test_top1:.4f}")
        if test_top1 > best_acc:
            best_model = student
            best_acc = test_top1

    torch.save(best_model.state_dict(), "models/best_acc_student_with_distillation.pth")
