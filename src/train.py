import hydra

from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from model.dynaformer import Dynaformer
from data_module.data_module import BatteryDataModule


@hydra.main(config_path="../config/", config_name="train")
def main(cfg):
    data_dir = Path(hydra.utils.to_absolute_path(cfg.data_dir))

    model = Dynaformer(
        lr=cfg.method.lr,
        patience_lr_plateau=cfg.patience_lr_plateau,
    )
    save_top_k = 1
    drop_final = True

    data = BatteryDataModule(
        data_dir=data_dir,
        type="single",
        batch_size=cfg.method.batch_size,
    )

    checkpoint_callback_pred = ModelCheckpoint(
        monitor="val_loss",
        dirpath="weights/",
        filename=cfg.method.name + "-{epoch:02d}-{prediction_error:.2f}",
        mode="min",
    )

    checkpoint_callback_train = ModelCheckpoint(
        monitor="train_loss",
        dirpath="weights/",
        filename=cfg.method.name + "_train_" + "-{epoch:02d}-{prediction_error:.2f}",
        mode="min",
        save_top_k=save_top_k,
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback_pred, checkpoint_callback_train],
        max_epochs=cfg.epochs,
        check_val_every_n_epoch=3,
        num_sanity_val_steps=0,
        accelerator="mps",
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
