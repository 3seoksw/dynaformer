import pickle
import os
import numpy as np

from preprocess.base import BasePreprocessor
from pathlib import Path
from tqdm import tqdm


class HUSTPreprocessor(BasePreprocessor):
    def __init__(self, discharge_type="single", data_dir="data/HUST"):
        self.discharge_type = discharge_type
        self.data_dir = data_dir
        self.length = 0
        self.discharges = []
        self.full_discharges = []
        
        if discharge_type == "single":
            self.preprocess_single()
        elif discharge_type == "full":
            self.preprocess_full()
        else:
            raise Exception(f"No such discharge type available: {discharge_type}")


    def __len__(self):
        return len(self.discharges)


    def save_to_file(self):
        dir_path = os.path.join(self.data_dir, self.discharge_type)
        os.makedirs(dir_path, exist_ok=True)

        for i, discharge in enumerate(self.discharges):
            with open(f"{dir_path}/discharge_{i}.pkl", "wb") as f:
                pickle.dump(discharge, f)


    def preprocess_single(self):
        """HUST dataset"""
        data_dir = os.path.join(self.data_dir, "our_data")
        data_dir = Path(data_dir)
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
    def preprocess_full(self):
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
