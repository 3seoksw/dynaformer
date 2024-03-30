from preprocess import *


TYPES = ["HUST"]


def main(dataset_type: str, data_dir: str, output_dir: str):
    assert dataset_type in TYPES
    if dataset_type == "HUST":
        preprocessor = HUSTPreprocessor(data_dir, output_dir)

    print()


if __name__ == "__main__":
    main()
