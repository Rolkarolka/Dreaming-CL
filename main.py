import os
from dataclasses import dataclass

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf, MISSING

from hydra.core.config_store import ConfigStore
from torch import nn
from torchvision import models

from data_module.DreamingDataModule import CIFARDataModule
from models.DreamingNet import DreamingNet
from utils.utils import visualize_output_space, embed_imgs


@dataclass
class ExperimentSchema:
    batch_size: int
    max_epochs: int
    name: str
    experiment_run: bool
    tracking_uri: str
    run_name: str


@dataclass
class ResnetExperimentSchema(ExperimentSchema):
    classes_to_learn: list[int]
    classes_to_dream: list[int]


@dataclass
class ConfigSchema:
    experiment: ExperimentSchema


cs = ConfigStore.instance()
cs.store(name="config_schema", node=ConfigSchema)
cs.store(group="experiment", name="resnet_schema", node=ResnetExperimentSchema)


def load_teacher_net(num_classes):
    teacher = models.resnet34()
    teacher_weights_path = os.path.join(os.getcwd(), "utils", 'teacher', 'teacher_resnet34_classes_0_1_2.weights')
    teacher_num_features = teacher.fc.in_features
    teacher.fc = nn.Linear(teacher_num_features, num_classes)
    teacher_checkpoint = torch.load(teacher_weights_path)
    teacher.load_state_dict(teacher_checkpoint)
    return teacher


def load_student_net(num_classes):
    student = models.resnet34()
    student_num_features = student.fc.in_features
    student.fc = nn.Linear(student_num_features, num_classes)
    return student

def visualize(mlf_logger, cifar_data_module, teacher, student):
    train_batch = cifar_data_module.train_dataloader()
    teacher_imgs, teacher_embeds, teacher_labels = embed_imgs(teacher, train_batch)
    student_imgs, student_embeds, student_labels = embed_imgs(student, train_batch)
    visualize_output_space(mlf_logger, teacher_imgs, teacher_embeds, teacher_labels, step="test_teacher")
    visualize_output_space(mlf_logger, student_imgs, student_embeds, student_labels, step="test_student")


def save(mlf_logger, dreaming_net):
    saved_students_weights = os.path.join(os.getcwd(), 'trained')
    os.makedirs(saved_students_weights, exist_ok=True)
    model_path = os.path.join(saved_students_weights, f"state_dict_model_{mlf_logger.run_id}.pt")
    torch.save(dreaming_net.student.state_dict(), model_path)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, model_path)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    experiment = config.experiment
    data_path = os.path.join(os.getcwd(), 'data')
    teacher = load_teacher_net(len(experiment.classes_to_dream))
    student = load_student_net(len(experiment.classes_to_dream) + len(experiment.classes_to_learn))

    mlf_logger = MLFlowLogger(experiment_name=experiment.name, tracking_uri=experiment.tracking_uri, run_name=experiment.run_name)
    dreaming_net = DreamingNet(teacher, student)
    cifar_data_module = CIFARDataModule(teacher, experiment.classes_to_learn, experiment.classes_to_dream, mlf_logger, data_path, experiment.batch_size)
    trainer = pl.Trainer(logger=mlf_logger if experiment.experiment_run else None, max_epochs=experiment.max_epochs)
    trainer.fit(model=dreaming_net, datamodule=cifar_data_module)

    visualize(mlf_logger, cifar_data_module, teacher, student)
    trainer.test(dreaming_net, datamodule=cifar_data_module)
    save(mlf_logger, dreaming_net)


if __name__ == '__main__':
    main()
