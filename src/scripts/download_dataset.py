import os
import requests

from pathlib import Path
from tqdm import tqdm


LINKS = {
    "HUST": [
        (
            "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/nsc7hnsg4s-2.zip"
        ),
    ],
    "RWTH": [
        ("https://publications.rwth-aachen.de/record/818642/files/Rawdata.zip"),
    ],
    "CALCE": [
        ("https://web.calce.umd.edu/batteries/data/CS2_33.zip"),
        ("https://web.calce.umd.edu/batteries/data/CS2_34.zip"),
        ("https://web.calce.umd.edu/batteries/data/CS2_35.zip"),
        ("https://web.calce.umd.edu/batteries/data/CS2_36.zip"),
        ("https://web.calce.umd.edu/batteries/data/CS2_37.zip"),
        ("https://web.calce.umd.edu/batteries/data/CS2_38.zip"),
        ("https://web.calce.umd.edu/batteries/data/CX2_16.zip"),
        ("https://web.calce.umd.edu/batteries/data/CX2_33.zip"),
        ("https://web.calce.umd.edu/batteries/data/CX2_35.zip"),
        ("https://web.calce.umd.edu/batteries/data/CX2_34.zip"),
        ("https://web.calce.umd.edu/batteries/data/CX2_36.zip"),
        ("https://web.calce.umd.edu/batteries/data/CX2_37.zip"),
        ("https://web.calce.umd.edu/batteries/data/CX2_38.zip"),
    ],
}


def main():
    DATA_DIR = "data"

    if not os.path.exists(DATA_DIR):
        # TODO: mkdir
        print("mkdir")

    for dataset in LINKS:
        data_dir = os.path.join(DATA_DIR, dataset)
        if os.path.exists(data_dir):
            # TODO: mkdir
            print(f"{dataset} already exists.")
            continue

        for url in LINKS[dataset]:
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise ValueError(
                    f"Network Response Value Error: {response.status_code} raised."
                )

            total_length = response.headers.get("content-length")

            # TODO: tqdm download
            print(total_length)
            print(f"{dataset}: {url}")


if __name__ == "__main__":
    main()
