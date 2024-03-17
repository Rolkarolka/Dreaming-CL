import numpy as np
import torch
from typing import Optional, Callable, List

from torchvision.datasets import CIFAR10
from PIL import Image


class CIFAR10Subset(CIFAR10):
    def __init__(self,
                 root: str,
                 all_classes: List[int],
                 classes_to_learn: List[int] = None,
                 dreamed_data=None,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 teacher_class_proportion = None,
                 download: bool = False):
        super().__init__(root=root, train=train, transform=transform, download=download)

        self.class_importance = self.get_class_importance(classes_to_learn, teacher_class_proportion)

        self.classes = [self.classes[i] for i in all_classes]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        if classes_to_learn and dreamed_data is not None:
            data_idx = [i for i, cls_idx in enumerate(self.targets) if cls_idx in classes_to_learn]
            self.targets = [self.targets[i] for i in data_idx] + dreamed_data.tensors[1].tolist()
            dreamed_imgs = dreamed_data.tensors[0].cpu().numpy().transpose([0, 2, 3, 1]).astype('uint8')
            self.data = np.concatenate((self.data[data_idx], dreamed_imgs))
        else:
            data_idx = [i for i, cls_idx in enumerate(self.targets) if cls_idx in all_classes]
            self.targets = [self.targets[i] for i in data_idx]
            self.data = self.data[data_idx]

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        importance = self.class_importance[target.item()]

        return img, target, importance

    def get_class_importance(self, classes_to_learn, teacher_class_proportion):
        samples_proportion = {}
        for class_name in classes_to_learn:
            print(self.targets)
            x = sum(torch.where(self.targets == class_name, 1, 0).item())
            print(x)
            samples_proportion[class_name] = sum(self.targets == class_name)
        if teacher_class_proportion is not None:
            samples_proportion.update(teacher_class_proportion)
        weights = {}
        all_samples = sum(samples_proportion.values())
        for class_name, num_samples in samples_proportion.items():
            weights = (all_samples - num_samples)/all_samples
        return weights
