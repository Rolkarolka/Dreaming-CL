import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from data_module.BalancedBatchSampler import BalancedBatchSampler
from data_module.CIFAR10Subset import CIFAR10Subset
from data_module.DeepInversion import DeepInversion


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, teacher, teacher_class_proportion, classes_to_learn, classes_to_dream, logger, data_dir: str = "./data",
                 batch_size: int = 32, num_workers: int = 7):
        super().__init__()
        self.logger = logger
        self.teacher = teacher
        self.classes_to_learn = classes_to_learn
        self.classes_to_dream = classes_to_dream
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_percentage = 0.8
        self.teacher_class_proportion = teacher_class_proportion

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
        samples_to_dream = self.get_amount_of_samples_to_dream()
        self.deep_inversion = DeepInversion(samples_to_dream, logger=self.logger)
        inversed_data = self.deep_inversion.run_inversion(self.teacher, self.classes_to_dream)

        self.train_val_data = CIFAR10Subset(root=self.data_dir, classes_to_learn=self.classes_to_learn,
                                            all_classes=self.all_classes, dreamed_data=inversed_data, train=True,
                                            download=True, teacher_class_proportion=self.teacher_class_proportion, transform=self.transform)

        self.weight_class_importance = self.get_class_importance()

        self.test_data = CIFAR10Subset(root=self.data_dir, all_classes=self.all_classes, train=False, download=True,
                                       transform=self.test_transform)
        self.test_sampler = BalancedBatchSampler(self.test_data, num_classes=len(self.all_classes),
                                                 batch_size=self.batch_size)

    def get_amount_of_samples_to_dream(self):
        max_number_of_dreamed_imgs = 1000
        samples_to_dream = {}
        for dreamed_class in self.teacher_class_proportion:
            num_samples = dreamed_class/sum(self.teacher_class_proportion.values()) * max_number_of_dreamed_imgs
            samples_to_dream[dreamed_class] = num_samples
        return samples_to_dream

    def get_class_importance(self):
        all_samples = sum(self.teacher_class_proportion.values()) + self.train_val_data


    def setup(self, stage: str):
        train_size = int(self.train_percentage * len(self.train_val_data))
        self.train_data, self.val_data = random_split(self.train_val_data,
                                                      (train_size, len(self.train_val_data) - train_size))
        self.train_sampler = BalancedBatchSampler(self.train_data, num_classes=len(self.all_classes),
                                                  batch_size=self.batch_size)
        self.val_sampler = BalancedBatchSampler(self.val_data, num_classes=len(self.all_classes),
                                                batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_sampler=self.train_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_sampler=self.val_sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_sampler=self.test_sampler, num_workers=self.num_workers)