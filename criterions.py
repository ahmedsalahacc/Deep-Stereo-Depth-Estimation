import torch

from torch import nn


class LogLoss(nn.Module):
    """Ensures that the loss is always positive and greater than 0"""

    def __init__(self):
        super(LogLoss, self).__init__()

    def forward(self, pred, target):
        pred_sum = torch.sum(pred)
        target_sum = torch.sum(target)
        bs = pred.shape[0]
        # calculate the mean log loss
        return (torch.log(pred_sum) - torch.log(target_sum)).abs() / bs


class DepthLoss(nn.Module):
    """Gets the depth loss"""

    def __init__(self):
        super(DepthLoss, self).__init__()
        self.log_loss = LogLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # get the log loss
        log_loss = self.log_loss(pred, target)
        # get the mse loss
        mse_loss = self.mse_loss(pred, target)
        # return the sum of the two losses
        return log_loss + mse_loss
