import matplotlib.pyplot as plt
import pickle

from model.dynaformer import Dynaformer
from data_module.data_module import BatteryDataModule
from data_module.HUST_dataset import HUSTBatteryDataset
from pathlib import Path
from pytorch_lightning import Trainer


def main():
    data_dir = "data/"

    model = Dynaformer()

    data = BatteryDataModule(
        discharge_type="single",
        data_dir=data_dir,
        dataset_name="HUST",
        batch_size=64,
        num_w=4,
    )

    trainer = Trainer(
        accelerator="mps",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=5,
        log_every_n_steps=1,
        max_epochs=10001,
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
