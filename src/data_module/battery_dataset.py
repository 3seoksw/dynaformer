from typing import Dict, List
from torch.utils.data import Dataset


class BatteryDataset(Dataset):
    """
    Base battery dataset class.
    Arguments:
        data_dir (string): path to the dataset
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def __len__(self) -> int:
        """returns size of the dataset"""
        return -1

    def __getitem__(self, idx) -> Dict[str, List[float]]:
        """returns i-th sample from the dataset such that `dataset[i]`."""
        return {"": [idx]}
