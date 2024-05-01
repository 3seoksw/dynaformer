import pickle
import os
import numpy as np
import pandas as pd

from preprocess.base import BasePreprocessor
from pathlib import Path
from tqdm import tqdm


class RWTHPreprocessor(BasePreprocessor):
    def __init__(self, discharge_type="single", data_dir="data/RWTH", run=True):
        self.run = run
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
        if not self.run:
            raise Exception("Testing mode")

        dir_path = os.path.join(self.data_dir, self.discharge_type)
        os.makedirs(dir_path, exist_ok=True)

        for i, discharge in enumerate(self.discharges):
            with open(f"{dir_path}/RWTH_{i}.pkl", "wb") as f:
                pickle.dump(discharge, f)

    def preprocess_single(self):
        """RWTH dataset"""
        data_dir = os.path.join(self.data_dir, "Rohdaten")
        data_dir = Path(data_dir)
        cells = [f"{i:003}" for i in range(2, 50)]
        cells = tqdm(
            cells,
            desc="Processing RWTH dataset ('single' mode)",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        for cell in cells:
            files = data_dir.glob(f"*{cell}=ZYK*Zyk*.csv")
            for file in files:
                df = pd.read_csv(file, skiprows=[1])
                df = df.drop_duplicates("Zeit").sort_values("Zeit")
                current = df["Strom"].values
                discharges = find_cycle_ends(current)

                count = 0
                for i in discharges:
                    count += 1
                    start, end = i
                    if start == 0 or start == 1:  # skip first outlier discharge graph
                        continue
                    discharge = df.iloc[start:end]

                    capacity = discharge["AhStep"].values.tolist()
                    voltage = discharge["Spannung"].values.tolist()
                    current = discharge["Strom"].abs().values.tolist()
                    time = discharge["Programmdauer"].values.tolist()
                    time_start = time[0]
                    time = [t - time_start for t in time]

                    discharge = (current, voltage, capacity, time)
                    interpolated_discharge = interpolate_data(discharge, length=4000)

                    self.discharges.append(interpolated_discharge)

    def preprocess_full(self):
        print()


def find_cycle_ends(current):
    cycle_end_indices = []

    start = 0
    end = 0
    for i in range(1, len(current)):
        prev = current[i - 1]
        curr = current[i]

        if prev == 0 and curr < 0:
            start = i
        if prev < 0 and curr == 0:
            end = i
            cycle_end_indices.append((start, end))

    return cycle_end_indices


def interpolate_data(discharge: tuple, length=4000):
    # TODO: Interpolation algorithm
    capacity, voltage, current, time = discharge
    assert (
        len(capacity) == len(voltage)
        and len(voltage) == len(current)
        and len(current) == len(time)
    )
    data_len = len(time)
    interp_factor = data_len / length

    interp_discharge = []
    for data in discharge:
        interp_data = np.interp(
            np.arange(0, data_len, interp_factor),
            np.arange(data_len),
            data,
        )
        interp_discharge.append(interp_data)

    return interp_discharge


if __name__ == "__main__":
    pre = RWTHPreprocessor()
    print(pre.__len__())
