#!/usr/bin/env python
import torch.nn.utils.prune as prune

from pruning_loop import PruningExperiment

experiment_number = 3

# Experiment 1: Random weights pruning
# Change amount = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
amount = 0.7
kwargs = {}
pruning_strategy_1 = [
    ("fc1", prune.random_unstructured, "weight", amount, kwargs),
    ("fc2", prune.random_unstructured, "weight", amount, kwargs),
    ("conv1", prune.random_unstructured, "weight", amount, kwargs),
    ("conv2", prune.random_unstructured, "weight", amount, kwargs),
]

# Experiment 2: Pruning based on norm
amount = 0.7
kwargs = {}
pruning_strategy_2 = [
    ("fc1", prune.l1_unstructured, "weight", amount, kwargs),
    ("fc2", prune.l1_unstructured, "weight", amount, kwargs),
    ("conv1", prune.l1_unstructured, "weight", amount, kwargs),
    ("conv2", prune.l1_unstructured, "weight", amount, kwargs),
]

# Experiment 3: Structural pruning with L1 norm
amount = 0.7
kwargs = {"n": 1, "dim": 0}

pruning_strategy_3 = [
    ("fc1", prune.ln_structured, "weight", amount, kwargs),
    ("conv1", prune.ln_structured, "weight", amount, kwargs),
    ("conv2", prune.ln_structured, "weight", amount, kwargs),
]


if experiment_number == 1:
    pe = PruningExperiment(
        pruning_strategy=pruning_strategy_1,
        epochs_prune_finetune=3,
        epochs_finetune=4,
        save_model="exp1",
    )

if experiment_number == 2:
    pe = PruningExperiment(
        pruning_strategy=pruning_strategy_2,
        epochs_prune_finetune=3,
        epochs_finetune=4,
        save_model="exp2",
    )

if experiment_number == 3:
    pe = PruningExperiment(
        pruning_strategy=pruning_strategy_3,
        epochs_prune_finetune=3,
        epochs_finetune=4,
        save_model="exp3",
    )


print(pe.run())
