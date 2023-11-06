import lightning.pytorch as pl
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, classes_to_learn, classes_to_dream, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.classes_to_learn = classes_to_learn
        self.classes_to_dream = classes_to_dream
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage: str):
        self.train_data = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transform)
        train_val_indices = [i for i in range(len(self.train_data)) if self.train_data[i][1] in self.classes_to_learn]

        train_size = int(0.8 * len(train_val_indices))
        self.train_indices = train_val_indices[:train_size]
        self.val_indices = train_val_indices[train_size:]

        self.test_data = datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.test_transform)
        classes_to_test = self.classes_to_learn + self.classes_to_dream
        self.test_indices = [i for i in range(len(self.test_data)) if self.test_data[i][1] in classes_to_test]

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, persistent_workers=True, num_workers=2,
                          sampler=SubsetRandomSampler(self.train_indices))

    def val_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, persistent_workers=True, num_workers=2,
                          sampler=SubsetRandomSampler(self.val_indices))

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, persistent_workers=True, num_workers=2,
                          sampler=SubsetRandomSampler(self.test_indices))

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, persistent_workers=True, num_workers=2,
                          sampler=SubsetRandomSampler(self.test_indices))
