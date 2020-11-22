import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim

from loaders import get_loaders
from maskedLayers import Conv2dMasked, LinearMasked
from metrics import flops, model_size


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2dMasked(1, 10, kernel_size=5)
        self.conv2 = Conv2dMasked(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = LinearMasked(320, 50)
        self.fc2 = LinearMasked(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, 1)


class PruningExperiment:
    def __init__(
        self,
        pruning_strategy=None,
        batch_size_train=512,
        batch_size_test=1024,
        epochs_finetune=10,
        device="cuda",
        save_model=None,
    ):

        """Initialize experiment
        :param pruning_strategy: Pruning strategy
        :param batch_size_train: Batch size for train
        :param batch_size_test: Batch size for test
        :param epochs_finetune: Number of epochs to finetune pruned model
        :param optimizer: Optimizer to perform gradient descent
        :param device: Device 'cpu' or 'cuda' where calculations are performed

        :return: Outcome of pruning strategy: Accuracy and pruning metrics
        """
        self.pruning_strategy = pruning_strategy
        self.device = device
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.epochs_finetune = epochs_finetune
        self.save_model = save_model

    def load_model(self):

        """Load LeNet model.
        All experiments will be performed on a trained model
        from the original script.
        """
        net = LeNet().to(self.device)
        net.load_state_dict(torch.load("models/best_acc.pth"))
        return net

    def prune_model(self, net, pruning_strategy):

        for modulename, strategy, name, amount in pruning_strategy:
            module = getattr(net, modulename)
            mask = strategy(module, name=name, amount=amount)
            module.set_mask(mask.weight_mask)
            prune.remove(module, name)
        return net

    def train(self, net, optimizer, data_loader, device):

        net.train()
        for (idx, (x, t)) in enumerate(data_loader):
            optimizer.zero_grad()
            x = net.forward(x.to(device))
            t = t.to(device)
            loss = F.nll_loss(x, t)
            loss.backward()
            optimizer.step()

    def test(self, net, data_loader, device):

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

    def calculate_prune_metrics(self, net, test_loader, device):

        x, _ = next(iter(test_loader))
        x = x.to(device)

        size, size_nz = model_size(net)

        FLOPS = flops(net, x)
        compression_ratio = size / size_nz
        return FLOPS, compression_ratio

    def run(self):
        """
        Main function to run pruning -> finetuning -> evaluation
        """
        pruning_strategy = self.pruning_strategy
        batch_size_train = self.batch_size_train
        batch_size_test = self.batch_size_test
        epochs_finetune = self.epochs_finetune
        device = self.device

        net = self.load_model()
        if pruning_strategy is not None:
            net = self.prune_model(net, pruning_strategy)
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

        train_loader, test_loader = get_loaders(batch_size_train, batch_size_test)

        test_top1 = self.test(net, test_loader, device)
        print(f"Categorical accuracy right after pruning is {test_top1}.\n")

        print("Finetuning pruned model")
        print("=======================")
        for epoch in range(epochs_finetune):
            self.train(net, optimizer, train_loader, device)
            test_top1 = self.test(net, test_loader, device)
            print(f"Epoch FineTuning {epoch}. Top1 {test_top1:.4f}")
        # module = net.fc1
        # print(torch.sum(torch.is_nonzero(net.fc1.weight)/np.prod(net.fc1.weight.shape))
        # print(list(module.named_parameters())[1])
        FLOPS, compression_ratio = self.calculate_prune_metrics(
            net, test_loader, device
        )

        if self.save_model is not None:
            torch.save(net.state_dict(), f"models/{self.save_model}.pth")

        return test_top1, FLOPS, compression_ratio
