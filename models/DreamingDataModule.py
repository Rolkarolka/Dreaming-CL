import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from models.BalancedBatchSampler import BalancedBatchSampler
from models.CIFAR10Subset import CIFAR10Subset
from models.DeepInversion import DeepInversion


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, teacher, classes_to_learn, classes_to_dream, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 7):
        super().__init__()
        self.teacher = teacher
        self.classes_to_learn = classes_to_learn
        self.classes_to_dream = classes_to_dream
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.all_classes = self.classes_to_learn + self.classes_to_dream
        # datasets
        class_num_samples = {class_name: 64 for class_name in self.classes_to_dream} # TODO 64 from config and depends on previous datasets
        self.deep_inversion = DeepInversion(class_num_samples)
        inversed_data = self.deep_inversion.run_inversion(self.teacher, self.classes_to_dream)
        self.train_val_data = CIFAR10Subset(root=self.data_dir, classes_to_learn=self.classes_to_learn, all_classes=self.all_classes, dreamed_data=inversed_data, train=True, download=True, transform=self.transform)

        self.test_data = CIFAR10Subset(root=self.data_dir, all_classes=self.all_classes, train=False, download=True, transform=self.test_transform)
        self.test_sampler = BalancedBatchSampler(self.test_data, num_classes=len(self.all_classes), batch_size=self.batch_size)

    def setup(self, stage: str):
        train_size = int(0.8 * len(self.train_val_data))
        self.train_data, self.val_data = random_split(self.train_val_data, (train_size, len(self.train_val_data) - train_size))
        self.train_sampler = BalancedBatchSampler(self.train_data, num_classes=len(self.all_classes), batch_size=self.batch_size)
        self.val_sampler = BalancedBatchSampler(self.val_data, num_classes=len(self.all_classes), batch_size=self.batch_size)


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_sampler=self.train_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_sampler=self.val_sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_sampler=self.test_sampler, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_sampler=self.test_sampler, num_workers=self.num_workers)
