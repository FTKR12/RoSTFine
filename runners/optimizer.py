import torch

def build_optim(optim_name, lr, momentum, weight_decay, model):
    # TODO: add optimizer
    optimizer_factory = {
        'SGD': torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
        'Adam': torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    }
    return optimizer_factory[optim_name]