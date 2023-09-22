import torch


def loss(loss_type: str):
    if loss_type == 'l1':
        loss_fn = torch.nn.L1Loss()
    elif loss_type == 'l2':
        loss_fn = torch.nn.MSELoss()
    return loss_fn
