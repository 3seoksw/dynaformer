import os
import pickle
import numpy as np
import json
import random

from data_module.battery_dataset import BatteryDataset
from pathlib import Path


def load_dataset(data_dir, split_val=False):
    data_dir = Path(data_dir)
    assert os.path.exists(data_dir / "metadata.json")
    with open(data_dir.joinpath("metadata.json"), "r") as f:
        metadata = json.load(f)

    if split_val:
        train_metadata = {}
        train_metadata["train_times"] = [
            x for x in range(metadata["train_times"]) if x % 25 != 0
        ]
        train_metadata["chunk_size"] = metadata["chunk_size"]
        train_metadata["data_dir"] = data_dir

        validation_metadata = {}
        validation_metadata["train_times"] = [
            x for x in range(metadata["train_times"]) if x % 25 == 0
        ]
        validation_metadata["chunk_size"] = metadata["chunk_size"]
        validation_metadata["data_dir"] = data_dir
        return train_metadata, validation_metadata
    else:
        train_metadata = {}
        train_metadata["train_times"] = [x for x in range(metadata["train_times"])]
        train_metadata["chunk_size"] = metadata["chunk_size"]
        train_metadata["data_dir"] = data_dir
        return train_metadata


class DynaBatteryDataset(BatteryDataset):
    def __init__(self, discharge_type: str, data_dir: str, mode: str):
        super().__init__(data_dir)

        self.mode = mode
        self.discharge_type = discharge_type  # "single"
        self.data_dir = os.path.join(
            data_dir, "variable_currents/2022-04-27/14-58-12/data"
        )

        self.requires_normalization = False
        self.is_single_query = False
        self.min_init = 0
        self.max_init = 50
        self.min_length = 200
        self.max_length = 200

        train_metadata, validation_metadata = load_dataset(
            self.data_dir, split_val=True
        )
        if self.mode == "train":
            self.metadata = train_metadata
            self.curves = self.metadata["train_times"]
        elif self.mode == "validation":
            self.metadata = validation_metadata
            self.curves = self.metadata["train_times"]
        elif self.mode == "test":
            self.metadata = validation_metadata
            self.curves = self.metadata["train_times"]
        else:
            raise KeyError("Mode error")

    def __len__(self):
        if self.mode == "train":
            return len(self.metadata["train_times"])
        elif self.mode == "val":
            return len(self.metadata["train_times"]) * 60
        elif self.mode == "test":
            return len(self.metadata["train_times"]) * 60
        else:
            raise KeyError()

    def __getitem__(self, idx):
        while True:
            if self.mode == "test":
                idx_curve = idx % len(self.curves)
            elif self.mode == "val":
                idx_curve = idx % len(self.curves)
            elif self.mode == "train":
                idx_curve = idx
            else:
                raise KeyError("mode must be either 'train' or 'test'")

            index = -1
            if self.mode == "test":
                index = self.metadata[f"test_times"][idx_curve]
            elif self.mode in ["train", "val"]:
                index = self.metadata[f"train_times"][idx_curve]
            file_idx = (index // self.metadata["chunk_size"]) * self.metadata[
                "chunk_size"
            ]
            sample_idx = index % self.metadata["chunk_size"]

            if self.mode == "test":
                current_path = (
                    self.metadata["data_dir"] / f"test_currentss_{file_idx}.pkl"
                )
                voltage_path = (
                    self.metadata["data_dir"] / f"test_voltages_{file_idx}.pkl"
                )
                times_path = self.metadata["data_dir"] / f"test_times_{file_idx}.pkl"
                q_path = self.metadata["data_dir"] / f"test_Qs_{file_idx}.pkl"
                r_path = self.metadata["data_dir"] / f"test_Rs_{file_idx}.pkl"
            else:
                current_path = (
                    self.metadata["data_dir"] / f"train_currents_{file_idx}.pkl"
                )
                voltage_path = (
                    self.metadata["data_dir"] / f"train_voltages_{file_idx}.pkl"
                )
                times_path = self.metadata["data_dir"] / f"train_times_{file_idx}.pkl"
                q_path = self.metadata["data_dir"] / f"train_Qs_{file_idx}.pkl"
                r_path = self.metadata["data_dir"] / f"train_Rs_{file_idx}.pkl"

            with open(current_path, "rb") as f:
                current_batch = pickle.load(f)

            with open(voltage_path, "rb") as f:
                voltage_batch = pickle.load(f)

            with open(times_path, "rb") as f:
                times_batch = pickle.load(f)

            if self.mode in set(["test", "val"]):
                with open(q_path, "rb") as f:
                    q_batch = pickle.load(f)

                with open(r_path, "rb") as f:
                    r_batch = pickle.load(f)

            voltage = voltage_batch[sample_idx]
            if self.mode == "train":
                # First index where voltage is lower than 3.2
                cut_off_idx = np.where(voltage <= 3.2)[0][0]

                if cut_off_idx < 300:
                    # Sample another
                    idx = random.randint(0, len(self.metadata["train_times"]) - 1)
                    continue
            break
        current = current_batch[sample_idx]
        voltage = voltage
        times = np.array(times_batch[sample_idx])
        current = np.array(
            current.get_current_profile(len(voltage) * 2)
        )  # TODO: FIXME when currents are variable #[:max_length]

        assert len(voltage) == len(current) == len(times)

        # Apply scalers if required
        if self.requires_normalization:
            # Apply MinMaxScaler to both voltage and current
            voltage = (voltage - 3.5) / 2
            current = (current - 2.5) / 2

        cut_off_idx = np.where(voltage <= 3.2)[0][0]
        if self.mode in ["test", "val"]:
            ratio = (idx // len(self.curves) + 70) / 100

            voltage = voltage[:cut_off_idx]
            gt_lenght = len(voltage)
            current = current[: len(voltage)]
            if ratio > 1:
                # assert not self.mode == "train"
                tmp = ratio - 1
                voltage = np.concatenate([voltage, np.zeros(int(len(voltage) * tmp))])
            else:
                voltage = voltage[: int(ratio * len(voltage))]
            if voltage.shape[0] > current.shape[0]:
                last_current_val = current[-1]
                current = np.concatenate(
                    [current, last_current_val * np.ones(len(voltage) - len(current))]
                )
            else:
                current = current[: len(voltage)]
        elif self.mode in ["train"]:

            # swapped because overwritten
            extendable_current = current[cut_off_idx:]
            extendable_voltage = voltage[cut_off_idx:]
            current = current[:cut_off_idx]
            voltage = voltage[:cut_off_idx]
            # Make sure that that the trajectory is longer at least 300 points considering ratio
            min_ratio = int(300 / len(voltage) * 100)
            if not self.is_single_query:
                ratio = random.randint(max(min_ratio, 55), 160) / 100
            else:
                ratio = 1.6  # Always use the maximum ratio
            if ratio > 1:
                # assert not self.mode == "train"
                tmp = ratio - 1
                concat_len = int(len(voltage) * tmp)
                to_add_voltage = extendable_voltage[:concat_len]
                to_add_current = extendable_current[:concat_len]
                voltage = np.concatenate([voltage, to_add_voltage])
                current = np.concatenate([current, to_add_current])
            else:
                new_len = int(len(voltage) * ratio)
                voltage = voltage[:new_len]
                current = current[:new_len]

            assert len(current) == len(voltage)
        else:
            raise KeyError()
        # scaled_current = scaled_current[:max_length]

        max_length = self.max_length
        min_length = self.min_length
        if max_length == min_length:
            length = max_length
        else:
            length = np.random.randint(min_length, max_length)

        min_init = self.min_init
        max_init = min(self.max_init, max(len(current) - length, 0))
        if max_init <= min_init:
            x_init = max_init
        else:
            x_init = np.random.randint(min_init, max_init)
        xx, yy, tt = (
            current[x_init : x_init + length],
            voltage[x_init : x_init + length],
            times[x_init : x_init + length],
        )
        if (not len(xx) == len(yy) == len(tt)) and self.mode == "val":
            return {}

        # if len(xx) == 0:
        #     breakpoint()

        datapoint = {}
        datapoint["current"] = current[x_init:]
        datapoint["voltage"] = voltage[x_init:]

        datapoint["xx"] = xx
        datapoint["yy"] = yy
        datapoint["tt"] = tt
        if self.mode in ["test", "val"]:
            datapoint["ratio"] = ratio
            datapoint["curve"] = sample_idx
            datapoint["dataset"] = file_idx
            datapoint["q"] = q_batch[sample_idx]
            datapoint["r"] = r_batch[sample_idx]
            datapoint["gt_length"] = gt_lenght

        # if (yy == 0).all():
        #     breakpoint()
        return datapoint
