import os

import torch
from lightning.pytorch.loggers import MLFlowLogger
import torch.nn as nn
from torchvision import models
import lightning.pytorch as pl

from models.DreamingDataModule import CIFARDataModule
from models.DreamingNet import DreamingNet

if __name__ == '__main__':
    batch_size = 64
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
    teacher_num_features = teacher.fc.in_features
    teacher.fc = nn.Linear(teacher_num_features, len(classes_to_dream))
    teacher_checkpoint = torch.load(teacher_weights_path)
    teacher.load_state_dict(teacher_checkpoint)

    # create student net
    student = models.resnet34()
    student_num_features = student.fc.in_features
    student.fc = nn.Linear(student_num_features, len(classes_to_dream) + len(classes_to_learn))

    # setup experiment
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri="databricks",
                              run_name=run_name)

    dreaming_net = DreamingNet(teacher, student)
    cifar_data_module = CIFARDataModule(teacher, classes_to_learn, classes_to_dream, data_path, batch_size)

    # train
    trainer = pl.Trainer(logger=mlf_logger if experiment_run else None, max_epochs=max_epochs)
    trainer.fit(model=dreaming_net, datamodule=cifar_data_module)

    # test
    trainer.test(dreaming_net, datamodule=cifar_data_module)

    # save
    model_path = os.path.join(saved_students_weights, f"state_dict_model_{mlf_logger.run_id}.pt")
    torch.save(dreaming_net.state_dict(), model_path)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, model_path)
