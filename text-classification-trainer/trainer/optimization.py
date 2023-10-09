import torch
import torch.nn as nn
import torch.optim


def get_loss():
    """
    Function to initialize cross entropy loss, optionally moving it to cuda if possible
    """
    
    loss  = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss = loss.cuda()
    
    return loss

def get_optimizer(
    model,
    optimizer = "SGD",
    learning_rate = 0.01,
    momentum = 0.5,
    weight_decay = 0,
):
    """
    Function to returnr an optimizer instance

    INPUT
    :model: the BERT model to optimize
    :optimizer: one of 'SGD' or 'Adam'
    :learning_rate: the learning rate
    :momentum: the momentum (if the optimizer uses it)
    :weight_decay: regularization coefficient
    
    OUTPUT
    :opt: optimizer instance
    """
    if optimizer.lower() == "sgd":
        opt = torch.optim.SGD(
            # YOUR CODE HERE
            model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )

    elif optimizer.lower() == "adam":
        opt = torch.optim.Adam(
            # YOUR CODE HERE
            model.parameters(), lr=learning_rate
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt