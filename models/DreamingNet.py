import lightning.pytorch as pl
import os

import torch
import csv
from torch import optim
import torch.nn as nn
import torchmetrics
from torchvision import models

from models.MetricLearningLoss import MetricLearningLoss
from utils.utils import embed_imgs, visualize_output_space
from utils.teacher.resnet_model import ResNet50


class DreamingNet(pl.LightningModule):
    def __init__(
            self,
            classes_to_dream,
            classes_to_learn,
            learning_rate=0.1
    ):
        super().__init__()
        self.teacher, self.teacher_class_proportion = self.load_teacher_net(len(classes_to_dream))
        self.student = self.load_student_net(len(classes_to_dream) + len(classes_to_learn))
        num_classes = self.student.fc.out_features
        self.loss_fun = nn.CrossEntropyLoss()
        self.metric_loss = MetricLearningLoss()
        self.learning_rate = learning_rate
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.student(x)
        loss = self.loss_fun(preds, y)
        loss += self.metric_loss(preds, y)
        acc = self.train_acc(preds, y)
        logs = {"train_loss": loss, "train_acc": acc}
        self.logger.log_metrics(logs, step=self.trainer.max_steps)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.student(x)
        loss = self.loss_fun(preds, y)
        acc = self.val_acc(preds, y)
        logs = {"val_loss": loss, "val_acc": acc}
        self.logger.log_metrics(logs, step=self.trainer.max_steps)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.student(x)
        acc = self.test_acc(preds, y)
        logs = {"test_acc": acc}
        self.logger.log_metrics(logs, step=self.trainer.max_steps)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_fit_start(self):
        self.logger.log_hyperparams({"learning rate": self.learning_rate, "loss function": self.loss_fun.__class__})
        # self.logger.log_graph(self.classifier)

    def load_teacher_net(self, num_classes):
        teacher = ResNet50(num_classes)
        teacher_weights_path = os.path.join(os.getcwd(), "utils", 'teacher', 'teacher_new_DataParallel_classes_0_1_2.weights')
        teacher_checkpoint = torch.load(teacher_weights_path)
        teacher.load_state_dict(teacher_checkpoint)

        teacher_class_proportion_path = os.path.join(os.getcwd(), "utils", 'teacher', 'dataset_info.csv')
        class_num_samples = self.csv_to_dict(teacher_class_proportion_path)
        return teacher, class_num_samples

    def csv_to_dict(self, file_path):
        data = {}
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data[int(row["class_name"])] = int(row["amount_samples"])
        return data

    def load_student_net(self, num_classes):
        student = ResNet50(num_classes)
        return student

    def visualize(self, cifar_data_module):
        train_batch = cifar_data_module.train_dataloader()
        teacher_imgs, teacher_embeds, teacher_labels = embed_imgs(self.teacher, train_batch)
        student_imgs, student_embeds, student_labels = embed_imgs(self.student, train_batch)
        visualize_output_space(self.logger, teacher_imgs, teacher_embeds, teacher_labels, step="train_teacher")
        visualize_output_space(self.logger, student_imgs, student_embeds, student_labels, step="train_student")

    def save(self):
        saved_students_weights = os.path.join(os.getcwd(), 'trained')
        os.makedirs(saved_students_weights, exist_ok=True)
        model_path = os.path.join(saved_students_weights, f"state_dict_model_{self.logger.run_id}.pt")
        torch.save(self.student.state_dict(), model_path)
        self.logger.experiment.log_artifact(self.logger.run_id, model_path)


