import os
import json
import torch
import numpy as np

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .battery_dataset import BatteryDataset
from .HUST_dataset import HUSTBatteryDataset


def collate_fn_padd(batch):
    """
    Collate batch in order to make the dataset within the batch in the same format (length)
    """

    max_current_length = max(len(b["current"]) for b in batch)
    max_voltage_length = max(len(b["voltage"]) for b in batch)
    assert max_current_length == max_voltage_length

    max_tt_length = max(len(b["tt"]) for b in batch)
    max_xx_length = max(len(b["xx"]) for b in batch)
    max_yy_length = max(len(b["yy"]) for b in batch)
    assert max_tt_length == max_xx_length and max_xx_length == max_yy_length

    padded_current = torch.tensor(
        np.array(
            [
                np.pad(b["current"], (0, max_current_length - len(b["current"])))
                for b in batch
            ]
        )
    ).float()
    padded_voltage = torch.tensor(
        np.array(
            [
                np.pad(b["voltage"], (0, max_voltage_length - len(b["voltage"])))
                for b in batch
            ]
        )
    ).float()
    padded_xx = torch.tensor(
        np.array([np.pad(b["xx"], (0, max_xx_length - len(b["xx"]))) for b in batch])
    ).float()
    padded_yy = torch.tensor(
        np.array([np.pad(b["yy"], (0, max_yy_length - len(b["yy"]))) for b in batch])
    ).float()
    padded_tt = torch.tensor(
        np.array([np.pad(b["tt"], (0, max_tt_length - len(b["tt"]))) for b in batch])
    ).float()

    return padded_current, padded_voltage, padded_xx, padded_yy, padded_tt


def load_dataset(data_dir, split=True):
    print(f"Opening {data_dir}")
    assert os.path.exists(data_dir / "metadata.json")
    with open(os.path.join(data_dir, "metadata.json"), "rb") as f:
        metadata = json.load(f)

    return metadata
    # if split:
    #     train_metadata = {}
    #     train_metadata["train_times"] =
    # else:
    #     train_metadata = {}


class BatteryDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        type,
        dataset_name="HUST",
        batch_size=64,  # In the original code, 12 or 64
        num_w=8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.type = type

        self.batch_size = batch_size
        self.num_w = num_w
        self.dataset_name = dataset_name

        if self.dataset_name == "HUST":
            self.training_dataset = HUSTBatteryDataset(
                type=self.type, data_dir=self.data_dir, mode="train"
            )

            self.validation_dataset = self.training_dataset
            self.validation_dataset.mode = "validation"

            self.test_dataset = self.training_dataset
            self.test_dataset.mode = "test"

        else:  # TODO: add more dataset compatibility
            self.training_dataset = BatteryDataset(data_dir=self.data_dir)

    def prepare_data(self):
        """Prepare the dataset trying to use"""

    def setup(self, stage: str):
        """
        Assign training, validation, testing datasets for use in dataloaders
        """
        if stage == "fit" or stage is None:
            self.shuffle = True
            self.drop_last = False
        else:
            self.shuffle = False
            self.drop_last = False

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_w,
            collate_fn=collate_fn_padd,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=self.drop_last,
            num_workers=self.num_w,
            collate_fn=collate_fn_padd,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=self.drop_last,
            num_workers=self.num_w,
            collate_fn=collate_fn_padd,
        )

    #
    # def predict_dataloader(self) -> DataLoader:
    #     return DataLoader(self.)
