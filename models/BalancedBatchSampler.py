import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, num_classes, batch_size):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_classes = num_classes
        self.indices_per_class = [[] for _ in range(num_classes)]

        # Organize indices by class
        for idx, (_, label) in enumerate(dataset):
            print(idx, label)
            self.indices_per_class[label].append(idx)

    def __iter__(self):
        while True:
            batch = []
            primary_class_name = np.random.randint(len(self.indices_per_class))
            primary_samples_size = self.batch_size//2 if len(self.indices_per_class[primary_class_name]) > self.batch_size//2 else len(self.indices_per_class[primary_class_name])
            batch += np.random.choice(self.indices_per_class[primary_class_name], primary_samples_size, replace=False).tolist()
            for _ in range((self.batch_size-primary_samples_size)//2):
                pair_class_name = np.random.randint(len(self.indices_per_class))
                batch += np.random.choice(self.indices_per_class[pair_class_name], 2, replace=True).tolist()
            yield iter(batch)

    def __len__(self):
        return len(self.dataset) // self.batch_size
