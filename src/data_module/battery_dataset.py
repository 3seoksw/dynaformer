import os
import pickle
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from preprocess.preprocess_HUST import HUSTPreprocessor
from preprocess.preprocess_RWTH import RWTHPreprocessor


class BatteryDataset(Dataset):
    """
    Base battery dataset class.
    Arguments:
        data_dir (string): path to the dataset
    """

    def __init__(
        self, data_dir: str, mode: str, discharge_type: str, dataset_name: str
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.discharge_type = discharge_type
        self.dataset_name = dataset_name

        preprocessed_data_dir = os.path.join(self.data_dir, discharge_type)
        if os.path.exists(preprocessed_data_dir):
            print("Preprocessed data loaded")
        else:
            print("Preprocess required!")

            if self.dataset_name == "HUST":
                preprocessor = HUSTPreprocessor(discharge_type, data_dir)
                preprocessor.save_to_file()
            elif self.dataset_name == "RWTH":
                preprocessor = RWTHPreprocessor(discharge_type, data_dir)
                preprocessor.save_to_file()
            elif self.dataset_name == "ALL":
                preprocessor = HUSTPreprocessor(discharge_type, data_dir)
                preprocessor.save_to_file()
                preprocessor = RWTHPreprocessor(discharge_type, data_dir)
                preprocessor.save_to_file()
            else:
                preprocessor = None
        self.data_dir = preprocessed_data_dir
        if self.dataset_name == "ALL":
            self.data_dir = "data/"

    def __len__(self) -> int:
        """returns size of the dataset"""
        if self.dataset_name == "ALL":
            length = 0
            dsets = ["HUST", "RWTH"]
            for dset in dsets:
                path = os.path.join(self.data_dir, dset)
                path = Path(path)
                files = list(path.glob("*.pkl"))
                length += len(files)

            return length

        files = list(Path(self.data_dir).glob("*.pkl"))
        return len(files)

    def __getitem__(self, idx):
        """returns i-th sample from the dataset such that `dataset[i]`."""
        if self.dataset_name == "ALL":
            discharge, cur_dset = self.__getdischarge__(idx)
        else:
            cur_dset = self.dataset_name
            with open(f"{self.data_dir}/{self.dataset_name}_{idx}.pkl", "rb") as f:
                discharge = pickle.load(f)

        current, voltage, capacity, time = discharge

        if self.discharge_type == "single":
            time_start = time[0]

            # NOTE: Context input is extracted at the point between 0s and 90s.
            max_context_start_idx = np.where(np.array(time) >= time_start + 90)[0][0]
            context_start_idx = np.random.randint(0, max_context_start_idx)
            time_start = time[context_start_idx]

            # NOTE: Fix context length to 400sec.
            # WARN: Since the length of HUST dataset is small, fix it to 100sec.
            if self.dataset_name == "HUST":
                context_end_idx = np.where(np.array(time) >= time_start + 100)[0][0]
            elif self.dataset_name == "RWTH":
                context_end_idx = 400

            # WARN: DEPRECATED
            # NOTE: Slice off the data when the battery discharges (3.2 V)
            # cut_off_list = np.where(np.array(voltage) <= 3.2)[0]
            # if len(cut_off_list) == 0:
            #     cut_off_idx = len(voltage)
            # else:
            #     cut_off_idx = cut_off_list[0]
            # full_current = current[context_start_idx:cut_off_idx]
            # full_voltage = voltage[context_start_idx:cut_off_idx]
            # full_capacity = capacity[context_start_idx:cut_off_idx]
            # full_time = time[context_start_idx:cut_off_idx]

            full_current = current[context_start_idx:]
            full_voltage = voltage[context_start_idx:]
            full_capacity = capacity[context_start_idx:]
            full_time = time[context_start_idx:]

            # status = status[context_start_idx:context_end_idx]
            context_current = current[context_start_idx:context_end_idx]
            context_voltage = voltage[context_start_idx:context_end_idx]
            # capacity = capacity[context_start_idx:context_end_idx]
            context_time = time[context_start_idx:context_end_idx]

            datapoint = {}

            # Context
            datapoint["xx"] = context_current
            datapoint["yy"] = context_voltage
            datapoint["tt"] = context_time

            # Full-length of current profile
            datapoint["current"] = full_current
            datapoint["voltage"] = full_voltage
            datapoint["time"] = full_time
            datapoint["capacity"] = full_capacity
            datapoint["metadata"] = {"dataset": cur_dset}

            return datapoint

        elif self.discharge_type == "full":
            return None

    def __getdischarge__(self, idx):
        length = 0
        dsets = {"HUST": 0, "RWTH": 0}
        cur_dset = ""
        for dset in dsets.keys():
            cur_dset = dset
            path = os.path.join(self.data_dir, dset)
            path = Path(path)
            files = list(path.glob("*.pkl"))
            length += len(files)
            dsets[dset] = length

            if idx <= length:
                break

        with open(f"{self.data_dir}/{cur_dset}/single/{cur_dset}_{idx}.pkl", "rb") as f:
            discharge = pickle.load(f)

        return discharge, cur_dset
