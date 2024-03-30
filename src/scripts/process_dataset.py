import hydra
import os
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


"""
https://data.mendeley.com/datasets/nsc7hnsg4s/2
77 cells w/ multiple charge-discharge cycles (#146,000)

Data Format:
1-1
    rul
        1504 data
    dq
        1504 data
    data
        1504 cycles
            ~= 800 charge-discharge per cycle
"""


# NOTE:
# TODO:
def trim_data(file_path):
    """Trims the data to contain one cycle"""
    print(file_path)
    return file_path


# TODO: finish the function
def format_file(file_path):
    """Write a new file to match the format of Dynaformer is using"""
    with open(file_path, "rb") as f:
        file = pickle.load(f)
        print(file)

    with open(f"../processed_data/train_times_{i}.pkl", "wb") as fp:
        pickle.dump(train_times[i : i + chunk_size], fp)


def process_data(data_dir=None):
    if data_dir is None:
        data_dir = "../data/nsc7hnsg4s-2/our_data/"

    print(len(os.listdir(data_dir)))
    for file_name in tqdm(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            file = trim_data(file_path)
            format_file(file)
        # format_file(file_path)


# @hydra.main(config_path="../config/", config_name="process")
def main():
    with open("../data/nsc7hnsg4s-2/our_data/1-1.pkl", "rb") as f:
        file = pickle.load(f)

    file = file["1-1"]
    rul_data = file["rul"]
    dq_data = file["dq"]
    data = file["data"]

    plt.title("Full cycles")
    plt.xlabel("Time")
    plt.ylabel("Capacity (Ah)")
    cycles = []
    capacities = []
    keys = range(1, 3)
    data = {k: data[k] for k in keys}
    print(len(data))
    for cycle in range(len(data)):
        c_data = data[cycle + 1]
        # c_data = c_data[c_data["Status"] != "Constant current charge"]
        c_data = c_data[c_data["Status"].str.contains("discharge")]

        idx = 0
        discharges = []
        indices = []
        cur_status = ""
        for i in range(len(c_data)):
            if cur_status != c_data["Status"].iloc[i]:
                cur_status = c_data["Status"].iloc[i]
                idx = c_data["Status"].iloc[i].split("_")[1]
                indices.append(i)
                discharges.append(idx)
        indices.append(len(c_data))

        # print(indices)
        discharges = []
        start = 0
        for i in range(len(indices) - 1):
            idx = indices[i]
            next_idx = indices[i + 1]

            status = c_data["Status"].iloc[idx:next_idx]
            current = c_data["Current (mA)"].iloc[idx:next_idx]
            voltage = c_data["Voltage (V)"].iloc[idx:next_idx]
            capacity = c_data["Capacity (mAh)"].iloc[idx:next_idx]
            time = c_data["Time (s)"].iloc[idx:next_idx]

            discharge = []
            discharges.append((status, current, voltage, capacity, time))
            # discharges.append(discharge)

        i = 0
        # print(discharges)
        for discharge in discharges:
            (_, _, voltage, capacity, time) = discharge
            print(time.iloc[1], time.iloc[0])
            plt.plot(time, capacity, label=f"{cycle}-{i}")
            i += 1

        # c_data = c_data[c_data["Status"] == "Constant current discharge_1"]
        # voltage = c_data["Voltage (V)"]
        # time = c_data["Time (s)"]
        # plt.plot(time, voltage)
        # cycles.append(cycle + 1)
        # capacity = c_data["Capacity (mAh)"] / 1000
        # capacity = capacity.mean()
        # capacities.append(capacity)

    # plt.plot(cycles, capacities)
    plt.legend()
    plt.show()

    data1 = data[data["Status"] == "Constant current discharge_1"]
    time1 = data1["Time (s)"]
    voltage1 = data1["Voltage (V)"]
    data3 = data[data["Status"] == "Constant current discharge_3"]
    time3 = data3["Time (s)"]
    voltage3 = data3["Voltage (V)"]

    if len(time1) >= len(time3):
        time = time3
        voltage1 = voltage1[: len(voltage3)]
    else:
        time = time1
        voltage3 = voltage3[: len(voltage1)]

    # plt.title("Discharge")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Voltage (V)")
    # plt.plot(time, voltage1, color="red")
    # plt.plot(time, voltage3, color="blue")

    """
    Status, Cycle number, Current (mA), Voltage (V), Capacity (mAh), Time (s)
    """
    # process_data()


if __name__ == "__main__":
    main()
