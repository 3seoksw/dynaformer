import matplotlib.pyplot as plt
import pickle

from model.dynaformer import Dynaformer
from data_module.data_module import BatteryDataModule
from data_module.HUST_dataset import HUSTBatteryDataset
from pathlib import Path
from pytorch_lightning import Trainer


def test():
    data = BatteryDataModule(data_dir="data/nsc7hnsg4s-2", type="single", batch_size=64)
    # test()
    # data = BatteryDataModule(data_dir="test/", type="single", batch_size=64)
    dataset = data.training_dataset

    for discharge in dataset:
        tt = discharge["tt"]
        yy = discharge["yy"]
        time = discharge["time"]
        voltage = discharge["voltage"]
        plt.plot(time, voltage, color="r", label="input")
        plt.plot(tt, yy, color="b", label="context")
        if len(tt) == 0 or len(time) == 0:
            print("error")

    plt.show()
    print(dataset.__len__())


def main():
    # data_dir = "data/nsc7hnsg4s-2"
    data_dir = "test/"

    model = Dynaformer()
    data = BatteryDataModule(data_dir=data_dir, type="single", num_w=4)
    trainer = Trainer(
        accelerator="mps",
        num_sanity_val_steps=0,
        # log_every_n_steps=10,
        log_every_n_steps=1,
        max_epochs=50,
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
