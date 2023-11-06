import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms

from models.CIFAR10Subset import CIFAR10Subset
from models.DeepInversion import DeepInversion


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, teacher, classes_to_learn, classes_to_dream, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.teacher = teacher
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

        # datasets
        self.cifar_data = CIFAR10Subset(root=self.data_dir, filtered_classes=self.classes_to_learn, train=True, download=True, transform=self.transform)
        # self.deep_inversion = DeepInversion(self.batch_size)
        # self.inversed_data = self.deep_inversion.run_inversion(self.teacher, self.classes_to_dream)
        # self.train_val_data = ConcatDataset([self.cifar_data, self.inversed_data])
        self.train_val_data = self.cifar_data

        classes_to_test = self.classes_to_learn + self.classes_to_dream
        self.test_data = CIFAR10Subset(root=self.data_dir, filtered_classes=classes_to_test, train=False, download=True, transform=self.test_transform)

    def setup(self, stage: str):
        train_size = int(0.8 * len(self.train_val_data))
        self.train_data, self.val_data = random_split(self.train_val_data, (train_size, len(self.train_val_data) - train_size))


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
