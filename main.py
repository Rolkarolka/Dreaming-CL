import os

import torch
import torchvision
from lightning.pytorch.loggers import MLFlowLogger
import torch.nn as nn
from torch.utils.data import Subset, SubsetRandomSampler
from torchvision import models
from torchvision.transforms import transforms
import lightning.pytorch as pl

from models.DreamingNet import DreamingNet

if __name__ == '__main__':
    batch_size = 64
    num_workers = 2
    max_epochs = 10
    classes_to_learn = [3, 4]  # during the next training round
    classes_to_dream = [0, 1, 2]  # what teacher already learned
    data_path = os.path.join(os.getcwd(), 'data')
    teacher_weights_path = os.path.join(os.getcwd(), 'teacher', 'teacher_resnet34_classes_0_1_2.weights')
    saved_students_weights = os.path.join(os.getcwd(), 'trained')
    experiment_name = "/Users/romanowskarolina@gmail.com/DreamingCL"
    tracking_uri = "databricks"
    run_name = input("Enter experiment run name: ")
    experiment_run = True
    if not run_name:
        experiment_run = False

    # load teacher net
    teacher = models.resnet34()
    num_ftrs = teacher.fc.in_features
    teacher.fc = nn.Linear(num_ftrs, len(classes_to_learn) + len(classes_to_dream))
    checkpoint = torch.load(teacher_weights_path)
    teacher.load_state_dict(checkpoint)
    dreaming_net = DreamingNet(teacher, len(classes_to_learn) + len(classes_to_dream))

    # load data
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    train_val_indices = [i for i in range(len(train_data)) if train_data[i][1] in classes_to_learn]

    train_size = int(0.8 * len(train_val_indices))
    train_indices = train_data[:train_size]
    val_indices = train_data[train_size:]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=2,
                                               sampler=SubsetRandomSampler(train_indices))
    val_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=2,
                                             sampler=SubsetRandomSampler(val_indices))

    test_data = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
    classes_to_test = classes_to_learn + classes_to_dream
    test_indices = [i for i in range(len(test_data)) if test_data[i][1] in classes_to_test]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers,
                                              sampler=SubsetRandomSampler(test_indices))

    # setup experiment
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri="databricks",
                              run_name=run_name)

    # train
    trainer = pl.Trainer(logger=mlf_logger if experiment_run else None, max_epochs=max_epochs)
    trainer.fit(model=dreaming_net, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # test
    trainer.test(dreaming_net, test_loader)

    # save
    model_path = os.path.join("trained", f"state_dict_model_{mlf_logger.run_id}.pt")
    torch.save(dreaming_net.state_dict(), model_path)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, model_path)
