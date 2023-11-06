from typing import Optional, Callable, List

from torchvision.datasets import CIFAR10


class CIFAR10Subset(CIFAR10):
    def __init__(self,
                 root: str,
                 filtered_classes: List[int],
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 download: bool = False):
        super().__init__(root=root, train=train, transform=transform, download=download)

        self.classes = [self.classes[i] for i in filtered_classes]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        data_idx = [i for i, cls_idx in enumerate(self.targets) if cls_idx in filtered_classes]
        self.targets = [self.targets[i] for i in data_idx]
        self.data = self.data[data_idx]
