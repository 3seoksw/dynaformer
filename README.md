> [!NOTE]
> Currently in progress of development!

# Dynaformer - Battery Discharge Prediction Model

This project is built based on the following paper: [[1]](#1).

## To Get Started

### Installation

```shell
conda create --name <environment> python=3.8.18
```

Replace `<environment>` with the name of virtual environment to your liking.

Then install the required packages as such:

```shell
conda activate <environment>
pip install -r requirements.txt
```

### Training

```shell
python src/train.py
```

<a id="1" href="https://www.sciencedirect.com/science/article/pii/S0306261923005937">[1]</a>
Luca Biggio, Tommaso Bendinelli, Chetan Kulkarni and Olga Fink. "Ageing-aware battery discharge prediction with deep learning"
<i>Applied Energy</i>, Volume. 346 (2023): 121229.
