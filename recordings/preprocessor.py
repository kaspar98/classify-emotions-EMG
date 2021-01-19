import os

import pandas as pd
import matplotlib.pyplot as plt


def clean():
    print("################################################################")
    print("Step 1 - Cleaning 'processed' folder of previous files.")
    current_path = os.path.dirname(os.path.abspath(__file__))
    processed_path = None

    for path in os.listdir(current_path):
        if not os.path.isdir(path):
            continue

        if "processed" not in path:
            continue
            
        processed_path = os.path.abspath(path)
        break
    
    for file_path in os.listdir(processed_path):
        os.remove(os.path.join(processed_path, file_path))
    
    print("Step 1 - Done")
    print("################################################################")
    print()


def read_data():
    print("################################################################")
    print("Step 2 - Collecting data from .txt files")

    datasets = {}

    raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
    for filepath in os.listdir(raw_path):
        print("Collecting data from: %s" % filepath)
        file_path = os.path.join(raw_path, filepath)
        current_data = pd.read_csv(file_path, sep=", ", header=0, skiprows=4, engine="python")
        current_data["Timestamp"] = pd.to_numeric(current_data["Timestamp"])

        datasets[filepath] = current_data

    print("Step 2 - Done")
    print("################################################################")
    print()

    return datasets


def clear_columns(datasets):
    retval = {}

    for key, value in datasets.items():
        retval[key] = value[["Sample Index", "EXG Channel 0", "EXG Channel 2", "Timestamp", "Timestamp (Formatted)"]]

    return retval


def clear_edges(datasets):
    retval = {}

    for key, value in datasets.items():
        try:
            start_seconds = float(key.split("_")[1].replace("start", "").replace(".txt", "").replace("s", ""))
            file_starttime_seconds = value["Timestamp"].iloc[0]

            file_correct_start = file_starttime_seconds + start_seconds
            file_correct_end = file_correct_start + 20 + 5 + 20 + 5 + 20

            retval[key] = value[(value["Timestamp"] > file_correct_start) & (value["Timestamp"] < file_correct_end)]
        except:
            print("SKIPPING %s: couldn't get starttime from filename. please use proper filename format 'name_startXs.txt' where X is the second slideshow happy started" % key)

    return retval


def print_file_lengths(datasets):
    for key, value in datasets.items():
        start = value["Timestamp"].iloc[0]
        end = value["Timestamp"].iloc[-1]

        print("File: %s, length: %s" % (key, str(end - start)))


def plot_datasets(datasets):
    count = 0
    
    for key, value in datasets.items():
        if count == 1:  # Only plot the first for now
            break
            
        ax = plt.gca()
        
        value.plot(kind="line", x="Timestamp", y="EXG Channel 0", ax=ax, color="blue")
        value.plot(kind="line", x="Timestamp", y="EXG Channel 2", ax=ax, color="red")

        plt.show()
        
        count += 1


def apply_bp_filter(datasets):
    retval = {}

    for key, value in datasets.items():
        retval[key] = value[(value["Sample Index"] >= 1.0) & (value["Sample Index"] <= 100.0) & value["Sample Index"] != 50.0]

    return retval


def run():
    clean()
    datasets = read_data()
    datasets = clear_columns(datasets)
    datasets = clear_edges(datasets)
    datasets = apply_bp_filter(datasets)

    # print_file_lengths(datasets)  # For debug to check if files are about the same length

    plot_datasets(datasets)

    print("Collected data for %i recordings" % len(datasets))


if __name__ == "__main__":
    run()
