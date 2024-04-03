import os
import pickle
import numpy as np

from data_module.battery_dataset import BatteryDataset
from pathlib import Path
from tqdm import tqdm


class HUSTBatteryDataset(BatteryDataset):
    def __init__(self, discharge_type: str, data_dir: str, mode: str):
        super().__init__(data_dir)

        self.length = 0

        self.data_dir = os.path.join(self.data_dir, "our_data")
        print(f"Opening {self.data_dir}...")
        assert os.path.exists(self.data_dir)

        self.discharges = []  # NOTE: for "single" type
        self.full_discharges = []  # NOTE: for "full" type

        self.mode = mode  # NOTE: mode = { "train", "validation", "test" }
        self.discharge_type = discharge_type

        if self.discharge_type == "single":
            self.process_single()
        elif self.discharge_type == "full":
            self.process_full()
        else:
            raise Exception(f"No such type available: {self.discharge_type}")

    def __len__(self):
        return len(self.discharges)

    # TODO: finish `__getitem__` function: `self.discharge_type == "full"`
    def __getitem__(self, idx):
        if self.discharge_type == "single":
            cur_discharge = self.discharges[idx]
            status, current, voltage, capacity, time = cur_discharge

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

    def process_single(self):
        data_dir = Path(self.data_dir)
        cell_files = list(data_dir.glob("*.pkl"))
        cell_files = tqdm(
            cell_files,
            desc="Processing HUST dataset ('single' mode)",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        for cell_file in cell_files:
            cell_id = cell_file.stem
            cell_name = f"HUST_{cell_id}"

            with open(cell_file, "rb") as cf:
                cell_data = pickle.load(cf)[cell_id]["data"]

            for cycle in range(len(cell_data)):
                cycle_data = cell_data[cycle + 1]

                # Extract discharge data exclusively
                cycle_data = cycle_data[cycle_data["Status"].str.contains("discharge")]

                # Save the indices when status changes
                # e.g.) Discharge1, Discharge1, Discharge2
                #           0            1           2
                #       indices = {2}
                indices = []
                cur_status = ""
                for i in range(len(cycle_data)):
                    if cur_status != cycle_data["Status"].iloc[i]:
                        cur_status = cycle_data["Status"].iloc[i]
                        indices.append(i)
                indices.append(len(cycle_data))

                # Save multiple single-discharge trajectories
                for i in range(len(indices) - 1):
                    idx = indices[i]
                    next_idx = indices[i + 1]

                    # HACK: This is to ensure sufficient context length. See `__getitem__()`.
                    # if next_idx - idx <= 90 // 4 + 100:
                    #     continue

                    status = cycle_data["Status"].iloc[idx:next_idx].values.tolist()
                    current = (
                        cycle_data["Current (mA)"].iloc[idx:next_idx].values.tolist()
                    )
                    voltage = (
                        cycle_data["Voltage (V)"].iloc[idx:next_idx].values.tolist()
                    )
                    capacity = (
                        cycle_data["Capacity (mAh)"].iloc[idx:next_idx].values.tolist()
                    )
                    time = cycle_data["Time (s)"].iloc[idx:next_idx].values.tolist()

                    time_start = time[0]
                    time = [t - time_start for t in time]

                    # HACK: Here, voltage data with too small values will be excluded;
                    # such as, voltage data containing less than 3.2 V
                    # NOTE: Context input is extracted between 0s and 90s.
                    # Time interval of HUST dataset is approximately 4sec.
                    max_context_start_idx = 90 // 4

                    # NOTE: Fix context length to 100 (equiv. to 400sec.):
                    max_context_end_idx = max_context_start_idx + 100

                    # NOTE: Slice off data when the battery discharges (3.2 V)
                    cut_off_list = np.where(np.array(voltage) <= 3.2)[0]
                    if len(cut_off_list) == 0:
                        cut_off_idx = len(voltage)
                    else:
                        cut_off_idx = cut_off_list[0]

                    if cut_off_idx <= max_context_end_idx:
                        continue

                    self.length += len(time)

                    discharge = (status, current, voltage, capacity, time)
                    self.discharges.append(discharge)

            cf.close()

    # TODO:
    def process_full(self):
        """
        Process full trajectory rather than single discharge trajectory.
        See `process_single` for comparison.
        """
        data_dir = Path(self.data_dir)
        cell_files = list(data_dir.glob(".pkl"))
        cell_files = tqdm(
            cell_files,
            desc="Processing HUST dataset ('full' mode)",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        for cell_file in cell_files:
            cell_id = cell_file.stem
            cell_name = f"HUST_{cell_id}"

            with open(cell_file, "rb") as cf:
                cell_data = pickle.load(cf)[cell_id]["data"]

            status = cell_data["Status"]
            current = cell_data["Current (mA)"]
            voltage = cell_data["Voltage (V)"]
            capacity = cell_data["Capacity (mAh)"]
            time = cell_data["Time (s)"]

            full_discharge = (status, current, voltage, capacity, time)
            self.full_discharges.append(full_discharge)

            cf.close()

        return False
