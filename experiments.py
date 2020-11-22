#!/usr/bin/env python
import torch.nn.utils.prune as prune

from pruning_loop import PruningExperiment

pruning_strategy = [
    ("fc1", prune.random_unstructured, "weight", 0.9),
    ("fc2", prune.random_unstructured, "weight", 0.9),
    ("conv1", prune.random_unstructured, "weight", 0.9),
    ("conv2", prune.random_unstructured, "weight", 0.9),
]

pe1 = PruningExperiment(
    pruning_strategy=pruning_strategy, epochs_finetune=1, save_model="exp1"
)

print(pe1.run())
