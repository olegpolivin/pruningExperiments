import torch


def calculate_global_sparsity(model):
    global_sparsity = (
        100.0
        * float(
            torch.sum(model.conv1.weight == 0)
            + torch.sum(model.conv2.weight == 0)
            + torch.sum(model.fc1.weight == 0)
            + torch.sum(model.fc2.weight == 0)
        )
        / float(
            model.conv1.weight.nelement()
            + model.conv2.weight.nelement()
            + model.fc1.weight.nelement()
            + model.fc2.weight.nelement()
        )
    )

    global_compression = 100 / (100 - global_sparsity)

    return global_sparsity, global_compression
