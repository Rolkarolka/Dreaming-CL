import torch
import torch.nn as nn


class MetricLearningLoss(nn.Module):
    def __init__(self, sigma=0.2, omega=1.0):
        super(MetricLearningLoss, self).__init__()
        self.sigma = sigma
        self.omega = omega

    def __call__(self, outputs, labels):
        loss = 0.0
        k = len(labels)
        # Always at least one pair from the same class
        print(f"Batch contains: {labels.unique(return_counts=True)}")
        print(labels)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                # Calculate Euclidean distance squared between outputs[i] and outputs[j]
                distance_squared = torch.sum((outputs[i] - outputs[j]) ** 2)

                # Check if the pair belongs to the same class or different classes
                if labels[i] == labels[j]:
                    loss += -(k / 2 - 1) * torch.log(distance_squared / (2 * k) / (self.sigma ** 2)) + 0.5 * (
                            distance_squared / (2 * k) / (self.sigma ** 2))
                else:
                    loss += (k / 2 - 1) * torch.log(distance_squared / (2 * k) / (self.omega ** 2)) - 0.5 * (
                            distance_squared / (2 * k) / (self.omega ** 2))

        return loss
