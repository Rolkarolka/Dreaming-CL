import numpy as np
from typing import Optional, Callable, List

from torchvision.datasets import CIFAR10


class CIFAR10Subset(CIFAR10):
    def __init__(self,
                 root: str,
                 all_classes: List[int],
                 classes_to_learn: List[int] = None,
                 dreamed_data=None,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 download: bool = False):
        super().__init__(root=root, train=train, transform=transform, download=download)

        self.classes = [self.classes[i] for i in all_classes]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print(self.data[0].shape)

        if classes_to_learn:
            data_idx = [i for i, cls_idx in enumerate(self.targets) if cls_idx in classes_to_learn]
            self.targets = [self.targets[i] for i in data_idx] + dreamed_data.tensors[1].tolist()
            dreamed_imgs = dreamed_data.tensors[0].cpu().numpy().transpose([0, 2, 3, 1]).astype('uint8')
            print(dreamed_imgs[0][0])
            self.data = np.concatenate((self.data[data_idx], dreamed_imgs))
        else:
            data_idx = [i for i, cls_idx in enumerate(self.targets) if cls_idx in all_classes]
            self.targets = [self.targets[i] for i in data_idx]
            self.data = self.data[data_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, batch_idx):
        return self.data[batch_idx], self.targets[batch_idx]
