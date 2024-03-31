import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule
from torch.optim import Adam


class BaseModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.tmp = []
        self.count = 0

    def training_step(self, batch, batch_idx):
        current, voltage, x, y, t = batch
        context = torch.cat([x.unsqueeze(2), y.unsqueeze(2), t.unsqueeze(2)], dim=2)
        output = self.forward(context, current)
        zero_mask = voltage != 0

        loss = self.loss_func(output[zero_mask].squeeze(), voltage[zero_mask].squeeze())

        if self.loss == "rmse":
            rmse_loss = loss
        else:
            raise KeyError("Loss function not recognized")

        # self.log("train_loss_rmse", rmse_loss, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.training_step_outputs.append(loss)
        self.tmp.append((output, voltage))

        return loss

    def on_train_epoch_end(self):
        epoch_avg = torch.stack(self.training_step_outputs).mean()
        self.log("train_epoch_end", epoch_avg)
        self.training_step_outputs.clear()

        if self.count % 500 == 0:
            output, voltage = self.tmp[-1]
            output = output.squeeze()
            output = output.cpu().detach().numpy()
            voltage = voltage.cpu().detach().numpy()
            output = output[-1, :]
            voltage = voltage[-1, :]

            # HACK: `cut_off_list` and `cut_off_idx` is to locate the index 
            # where zero padding is applied
            cut_off_list = np.where(np.array(voltage) <= 0)[0]

            # no padding found: able to use the original
            if len(cut_off_list) == 0:
                cut_off_idx = len(voltage)
            else:
                cut_off_idx = cut_off_list[0]

            output = output[:cut_off_idx]
            voltage = voltage[:cut_off_idx]

            plt.title(self.global_step)
            plt.xlabel("Time (4s)")
            plt.ylabel("Voltage (V)")
            plt.plot(output, "r--", label="predicted traj")
            plt.plot(voltage, "b-", label="actual traj")
            plt.legend()

            plt.savefig(f"{self.global_step}-pred.png")
            plt.close()

        self.count += 1

        self.tmp.clear()

    def validation_step(self, batch, batch_idx):
        current, voltage, x, y, t = batch
        context = torch.cat([x.unsqueeze(2), y.unsqueeze(2), t.unsqueeze(2)], dim=2)
        output = self.forward(context, current)
        zero_mask = voltage != 0

        # TODO: currently no metadata available

        loss = self.loss_func(output[zero_mask].squeeze(), voltage[zero_mask].squeeze())
        self.log("validation_loss", loss, on_step=True, on_epoch=True)
        self.validation_step_outputs.append(loss)

        return loss

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_end", epoch_average)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        current, voltage, x, y, t = batch
        context = torch.cat([x.unsqueeze(2), y.unsqueeze(2), t.unsqueeze(2)], dim=2)
        output = self.forward(context, current)
        zero_mask = voltage != 0

        self.test_step_outputs.append((output, voltage))

    def on_test_epoch_end(self):
        output, voltage = self.test_step_outputs[-1]
        self.log("predicted_traj", output)
        self.log("actual_traj", voltage)

    def configure_optimizers(self):
        """defines model optimizer"""
        opt = Adam(self.parameters(), lr=self.lr)
        lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self.patience_lr_plateau
        )
        return {
            "optimizer": opt,
            "lr_scheduler": lr_schedulers,
            "monitor": "train_loss",
        }
