import torch
import torch.nn as nn


class MetricLearningLoss(nn.Module):
    def __init__(
        self,
        sigma=0.2,
        omega=1.0
    ):
        super(MetricLearningLoss, self).__init__()
        self.sigma = sigma
        self.omega = omega

    def __call__(self, outputs, labels):
        k = len(labels)
        distance_sq = torch.cdist(outputs, outputs, p=2)**2
        same_class_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        same_class_loss = torch.sum(-(k / 2 - 1) * torch.log(distance_sq / (2 * k) / (self.sigma ** 2)) + 0.5 * (
                distance_sq / (2 * k) / (self.sigma ** 2)) * same_class_mask)
        different_class_loss = torch.sum((k / 2 - 1) * torch.log(distance_sq / (2 * k) / (self.omega ** 2)) - 0.5 * (
                distance_sq / (2 * k) / (self.omega ** 2)) * ~same_class_mask)

        return same_class_loss + different_class_loss
