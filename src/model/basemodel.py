import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

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
        current, voltage, capacity, x, y, t = batch
        context = torch.cat([x.unsqueeze(2), y.unsqueeze(2), t.unsqueeze(2)], dim=2)
        output, encoder = self.forward(context, current)
        zero_mask = voltage != 0

        loss = self.loss_func(output[zero_mask].squeeze(), voltage[zero_mask].squeeze())
        loss = torch.sqrt(loss)

        if math.isnan(loss):
            print(voltage.shape)
            voltage = voltage.cpu().detach().numpy()
            print(voltage, voltage.shape)
            print(current, current.shape)
            plt.plot(voltage[0])
            plt.show()
            exit()

        if self.loss == "rmse":
            rmse_loss = loss
        else:
            raise KeyError("Loss function not recognized")

        # self.log("train_loss_rmse", rmse_loss, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True)
        self.training_step_outputs.append(loss)
        self.tmp.append((output, voltage))

        return loss

    def on_train_epoch_end(self):
        epoch_avg = torch.stack(self.training_step_outputs).mean()
        self.log("train_epoch_end", epoch_avg)
        self.training_step_outputs.clear()

        if self.count % 100 == 0:
            output, voltage = self.tmp[-1]
            output = output.squeeze()
            output = output.cpu().detach().numpy()
            voltage = voltage.cpu().detach().numpy()

            batch_size = output.shape[0]

            if self.count % 500 == 0:
                for i in range(batch_size):
                    o = output[i, :]
                    v = voltage[i, :]
                    draw_trajectory(o, v, self.count, True, i)
            else:
                o = output[-1, :]
                v = voltage[-1, :]
                draw_trajectory(o, v, self.count)

        self.count += 1

        self.tmp.clear()

    def validation_step(self, batch, batch_idx):
        current, voltage, capacity, x, y, t = batch
        context = torch.cat([x.unsqueeze(2), y.unsqueeze(2), t.unsqueeze(2)], dim=2)
        output, encoder = self.forward(context, current)
        zero_mask = voltage != 0

        loss = self.loss_func(output[zero_mask].squeeze(), voltage[zero_mask].squeeze())
        self.log("validation_loss", loss, on_step=True)
        self.validation_step_outputs.append(loss)

        return loss

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_end", epoch_average)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        current, voltage, capacity, x, y, t = batch
        context = torch.cat([x.unsqueeze(2), y.unsqueeze(2), t.unsqueeze(2)], dim=2)
        output, encoder = self.forward(context, current)
        zero_mask = voltage != 0

        self.test_step_outputs.append((output, voltage))

    def on_test_epoch_end(self):
        output, voltage = self.test_step_outputs[-1]

        output = output.squeeze()
        output = output.cpu().detach().numpy()
        voltage = voltage.cpu().detach().numpy()

        batch_size = output.shape[0]

        if self.count % 10 == 0:
            for i in range(batch_size):
                o = output[i, :]
                v = voltage[i, :]
                draw_trajectory(o, v, self.count, True, i)
        else:
            o = output[-1, :]
            v = voltage[-1, :]
            draw_trajectory(o, v, self.count)

        # ====================== #
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
            "lr_scheduler": {
                "scheduler": lr_schedulers,
                "monitor": "train_loss"
            }
        }


def draw_trajectory(output, voltage, count, is_batch=False, num=63):
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

    plt.title(count)
    plt.xlabel("Time (4s)")
    plt.ylabel("Voltage (V)")
    plt.plot(output, "r--", label="predicted traj")
    plt.plot(voltage, "b-", label="actual traj")
    plt.legend()

    if is_batch:
        dir_path = os.path.join("pics", f"{count}")
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(f"pics/{count}/{count}-{num}-pred.png")
    else:
        plt.savefig(f"pics/{count}-{num}-pred.png")

    plt.close()
