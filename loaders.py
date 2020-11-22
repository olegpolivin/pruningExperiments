import torch
import torchvision


def get_loaders(batch_size_train, batch_size_test):
    """Function to return train and test datasets for MNIST

    :param batch_size_train: Batch size used for train
    :param batch_size_test: Batch size used for test

    :return: Data loaders for train and test data
    """

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

    return train_loader, test_loader
