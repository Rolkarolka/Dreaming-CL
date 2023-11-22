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
            self.indices_per_class[label].append(idx)

    def __iter__(self):
        batch = []
        samples_from_class = self.batch_size // self.num_classes

        for class_indices in self.indices_per_class:
            chosen_indices = np.random.choice(class_indices, samples_from_class, replace=False)
            batch.append(chosen_indices)
            # for i in sorted(chosen_indices.tolist(), reverse=True):
            #     del class_indices[i]

        num_random_samples = self.batch_size - len(batch)
        print(self.batch_size, len(batch), num_random_samples)
        if num_random_samples > 0:
            chosen_classes = np.random.choice(self.num_classes, num_random_samples, replace=True)
            for class_idx in chosen_classes:
                batch.append(self.indices_per_class[class_idx][0])
                # del self.indices_per_class[class_idx][0]
        return batch

    def __len__(self):
        return len(self.dataset) // self.batch_size
