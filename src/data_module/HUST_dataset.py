import os
import pickle
import numpy as np

from preprocess.preprocess_HUST import HUSTPreprocessor
from data_module.battery_dataset import BatteryDataset
from pathlib import Path
from tqdm import tqdm


class HUSTBatteryDataset(BatteryDataset):
    def __init__(self, discharge_type: str, data_dir: str, mode: str):
        super().__init__(data_dir)

        self.length = 0

        preprocessed_data_dir = os.path.join(self.data_dir, discharge_type)
        if os.path.exists(preprocessed_data_dir):
            print(f"Preprocessed data loaded")
        else:
            print(f"Preprocess required!")
            preprocessor = HUSTPreprocessor(discharge_type=discharge_type, data_dir=data_dir)
            preprocessor.save_to_file()

        self.data_dir = preprocessed_data_dir

        self.discharges = []  # NOTE: for "single" type
        self.full_discharges = []  # NOTE: for "full" type

        self.mode = mode  # NOTE: mode = { "train", "validation", "test" }
        self.discharge_type = discharge_type


    def __len__(self):
        files = list(Path(self.data_dir).glob("*.pkl"))
        return len(files)


    def __getitem__(self, idx):
        with open(f"{self.data_dir}/discharge_{idx}.pkl", "rb") as f:
            discharge = pickle.load(f)

        status, current, voltage, capacity, time = discharge

        if self.discharge_type == "single":
            time_start = time[0]

            # NOTE: Context input is extracted at the point between 0s and 90s.
            max_context_start_idx = np.where(np.array(time) >= time_start + 90)[0][0]
            context_start_idx = np.random.randint(0, max_context_start_idx)
            # Time interval of HUST dataset is approximately 4sec.
            # context_start_idx = np.random.randint(0, 90 // 4)
            time_start = time[context_start_idx]

            # NOTE: Fix context length to 100 (equiv. to 400sec.):
            # WARN: Since the length of HUST dataset is small, fix it to 100sec.
            context_end_idx = np.where(np.array(time) >= time_start + 100)[0][0]
            # context_end_idx = context_start_idx + 100 // 4

            # NOTE: Slice off the data when the battery discharges (3.2 V)
            cut_off_list = np.where(np.array(voltage) <= 3.2)[0]
            if len(cut_off_list) == 0:
                cut_off_idx = len(voltage)
            else:
                cut_off_idx = cut_off_list[0]

            # full_status = status[context_start_idx:cut_off_idx]
            full_current = current[context_start_idx:cut_off_idx]
            full_voltage = voltage[context_start_idx:cut_off_idx]
            # full_capacity = capacity[context_start_idx:cut_off_idx]
            full_time = time[context_start_idx:cut_off_idx]

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

            return datapoint

        elif self.discharge_type == "full":
            cur_discharge = self.full_discharges[idx]
            status, current, voltage, capacity, time = cur_discharge

            context_len = 90
            time_start = time[0]

            max_context_start_idx = np.where(
                np.array(time) >= time_start + context_len
            )[0][0]
            context_start_idx = np.random.randint(0, max_context_start_idx)
            # Time interval of HUST dataset is approximately 4sec.
            # context_start_idx = np.random.randint(0, 90 // 4)
            time_start = time[context_start_idx]

            # NOTE: Fix context length to 100 (equiv. to 400sec.):
            # WARN: Since the length of HUST dataset is small, fix it to 100sec.
            context_end_idx = np.where(np.array(time) >= time_start + 100)[0][0]
            # context_end_idx = context_start_idx + 100 // 4

            # NOTE: Slice off the data when the battery discharges (3.2 V)
            cut_off_list = np.where(np.array(voltage) <= 3.2)[0]

            # TODO: set context length (90sec for single discharge process)

            datapoint = {}

            return self.full_discharges[idx]
