import lightning.pytorch as pl
from torch import optim
import torch.nn as nn
import torchmetrics


class DreamingNet(pl.LightningModule):
    def __init__(self, teacher, num_classes, learning_rate=0.1):
        super().__init__()
        self.teacher = teacher
        self.loss_fun = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.teacher(x)
        loss = self.loss_fun(preds, y)
        acc = self.train_acc(preds, y)
        logs = {"train_loss": loss, "train_acc": acc}
        self.logger.log_metrics(logs, step=self.trainer.max_steps)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.teacher(x)
        loss = self.loss_fun(preds, y)
        acc = self.val_acc(preds, y)
        logs = {"val_loss": loss, "val_acc": acc}
        self.logger.log_metrics(logs, step=self.trainer.max_steps)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds= self.teacher(x)
        acc = self.test_acc(preds, y)
        logs = {"test_acc": acc}
        self.logger.log_metrics(logs, step=self.trainer.max_steps)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_fit_start(self):
        self.logger.log_hyperparams({"learning rate": self.learning_rate, "loss function": self.loss_fun.__class__})
        # self.logger.log_graph(self.classifier)


