import pickle

from data_module.battery_dataset import BatteryDataset
from data.battery_data import BatteryData
from tqdm import tqdm


class SingleCycleBatteryDataset(BatteryDataset):
    """
    Battery dataset class for testing a single discharge cycle
    This is called when the `type` is equal to "single".
    """

    def __init__(self, type: str, data_dir: str):
        super().__init__(data_dir)

        self.type = type
