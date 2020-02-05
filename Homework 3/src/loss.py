import torch
import torch.nn as nn


class TripletAndPairLoss(nn.Module):
    def __init__(self, batch_size: int, margin: float = 0.01, pair_loss_weight_factor: float = 1):
        super(TripletAndPairLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = torch.tensor(margin)
        self.pair_loss_weight_factor = pair_loss_weight_factor

    def forward(self, x):
        anchor_end = self.batch_size
        positive_end = self.batch_size * 2
        anchor = x[0:anchor_end]
        positive = x[anchor_end:positive_end]
        negative = x[positive_end:]
        positive_distance: torch.Tensor = torch.sum((anchor - positive).pow(2), dim=1)
        negative_distance: torch.Tensor = torch.sum((anchor - negative).pow(2), dim=1)
        triplet_loss = torch.max(torch.tensor(0.0), (torch.tensor(1.0) - negative_distance / (positive_distance + self.margin)))
        pair_loss = positive_distance

        triplet_loss = torch.sum(triplet_loss)
        pair_loss = self.pair_loss_weight_factor * torch.sum(pair_loss)
        return triplet_loss, pair_loss
