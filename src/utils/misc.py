from typing import List

from torch.optim.optimizer import Optimizer
import torch.nn as nn
import numpy as np


def stepwise_learning_rate_decay(optimizer: Optimizer, learning_rate: float, iteration_number: int,
                                 steps: List, reduce: float = 0.1) -> float:
    if iteration_number in steps:
        steps.remove(iteration_number)
        learning_rate *= reduce
        print("Reduce learning rate to {}".format(learning_rate))

        for param in optimizer.param_groups:
            param["lr"] = learning_rate

    return learning_rate


def num_model_weights(model: nn.Module) -> int:
    num_weights = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    return num_weights
