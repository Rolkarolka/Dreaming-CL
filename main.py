import os
from dataclasses import dataclass

import hydra
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig
from typing import List

from hydra.core.config_store import ConfigStore

from data_module.DreamingDataModule import CIFARDataModule
from models.DreamingNet import DreamingNet


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
    classes_to_learn: List[int]
    classes_to_dream: List[int]


@dataclass
class ConfigSchema:
    experiment: ExperimentSchema


cs = ConfigStore.instance()
cs.store(name="config_schema", node=ConfigSchema)
cs.store(group="experiment", name="resnet_schema", node=ResnetExperimentSchema)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    experiment = config.experiment
    data_path = os.path.join(os.getcwd(), 'data')

    mlf_logger = MLFlowLogger(experiment_name=experiment.name, tracking_uri=experiment.tracking_uri,
                              run_name=experiment.run_name)
    dreaming_net = DreamingNet(experiment.classes_to_dream, experiment.classes_to_learn)
    cifar_data_module = CIFARDataModule(dreaming_net.teacher, dreaming_net.teacher_class_proportion,
                                        experiment.classes_to_learn, experiment.classes_to_dream, mlf_logger, data_path,
                                        experiment.batch_size)
    trainer = pl.Trainer(logger=mlf_logger if experiment.experiment_run else None, max_epochs=experiment.max_epochs)
    trainer.fit(model=dreaming_net, datamodule=cifar_data_module)

    dreaming_net.visualize(cifar_data_module)
    trainer.test(dreaming_net, datamodule=cifar_data_module)
    dreaming_net.save()


if __name__ == '__main__':
    main()
