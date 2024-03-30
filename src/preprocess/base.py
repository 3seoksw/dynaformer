from tqdm import tqdm


class BasePreprocessor:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir

    def preprocess(self):
        """Preprocessing according to its dataset type"""

    def dump(self, batteries):
        batteries = tqdm(batteries, desc=f"Dumping batteries to {str(self.output_dir)}")
        for battery in batteries:
            battery.dump(self.output_dir / f"{battery.cell_id}.pkl")

    def __call__(self):
        batteries = self.preprocess()
        self.dump(batteries)
