import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from scipy import signal

from features import SignalFeatures

"""
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
"""

def read_data():
    print("################################################################")
    print("Step 1 - Collecting data from .txt files")

    datasets = {}

    raw_path = os.path.join(os.getcwd(), "raw")
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

    fs = 250 # sampling frequency

    for key, value in datasets.items():
        try:
            start_seconds = int(key.split("_")[1].replace("start", "").replace(".txt", "").replace("s", ""))

            # timestamps are not accurate, frequency should be https://github.com/OpenBCI/OpenBCI_GUI/issues/129
            file_correct_start = fs * start_seconds
            file_correct_end = file_correct_start + fs * (20 + 5 + 20 + 5 + 20)

            retval[key] = value[file_correct_start:file_correct_end]
        except:
            print("SKIPPING %s: couldn't get starttime from filename. please use proper filename format 'name_startXs.txt' where X is the second slideshow happy started" % key)

    return retval


def print_file_lengths(datasets):
    for key, value in datasets.items():

        start = value["Timestamp"].iloc[0]
        end = value["Timestamp"].iloc[-1]

        print(f"File: {key}, length: {end - start}, samples {len(value)}")

# Method from https://github.com/J77M/openbciGui_filter_test/blob/master/gui_saved_data_filter.ipynb?fbclid=IwAR2_1W5Gzs2n57qsdpTSVqSsj6NBnRrWcqbcYmwTx8syKcISJosmrZkDHFg
def bandpass(start, stop, data, fs=250):
    bp_Hz = np.array([start, stop])
    b, a = signal.butter(5, bp_Hz / (fs / 2.0), btype='bandpass')
    return signal.lfilter(b, a, data, axis=0)

# Method from https://github.com/J77M/openbciGui_filter_test/blob/master/gui_saved_data_filter.ipynb?fbclid=IwAR2_1W5Gzs2n57qsdpTSVqSsj6NBnRrWcqbcYmwTx8syKcISJosmrZkDHFg
def notch(val, data, fs=250):
    notch_freq_Hz = np.array([float(val)])
    for freq_Hz in np.nditer(notch_freq_Hz):
        bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])
        b, a = signal.butter(3, bp_stop_Hz / (fs / 2.0), 'bandstop')
        fin = data = signal.lfilter(b, a, data)
    return fin

def apply_bp_filter(datasets):
    retval = {}

    band_low_value = 1.0
    band_high_value = 100.0
    notch_value = 50.0

    for key, current_df in datasets.items():

        for channel in [0, 2]:
            raw_data = list(current_df["EXG Channel %s" % str(channel)])
            notched_data = notch(notch_value, raw_data)
            bp_data = bandpass(band_low_value, band_high_value, notched_data)

            current_df["EXG Channel %s" % str(channel)] = bp_data

        retval[key] = current_df

    return retval


def normalize_data(datasets):

    retval = {}

    for key, current_df in datasets.items():

        for channel in [0, 2]:

            selector = "EXG Channel %s" % str(channel)
            data = current_df[selector]
            # 'data - mean' to center values around 0
            # 'data/max' to push all values between -1..1
            current_df[selector] = (data - data.mean()) / data.abs().max()

        retval[key] = current_df

    return retval


def extract_picture_blocks(datasets):

    def cut_ends(data, fs=250, length=2):
        return data[length*fs:len(data)-length*fs]

    picture_blocks = defaultdict(lambda: [])

    fs = 250 # sampling frequency

    for recording_name, current_df in datasets.items():

        for channel in [0, 2]:

            data = current_df["EXG Channel %s" % str(channel)]

            happy   = cut_ends(data[0*fs  : 20*fs])
            neutral = cut_ends(data[25*fs : 45*fs])
            sad     = cut_ends(data[50*fs : 70*fs])

            picture_blocks[("happy", channel, recording_name)].append(happy)
            picture_blocks[("neutral", channel, recording_name)].append(neutral)
            picture_blocks[("sad", channel, recording_name)].append(sad)

    return picture_blocks


def extract_data_samples(picture_blocks, sample_length, sample_step):
    """ Extract one sample every sample_step (s) with sample_length (s)
    """

    def channel_nr_to_muscle(channel_nr):
        if channel_nr == 0:
            return "Zygomaticus Major"
        elif channel_nr == 2:
            return "Corrugator Supercilii"

    def recording_name_to_person(recording_name):
        return recording_name.split("_")[0]

    fs = 250 # sampling_frequency
    data_samples = defaultdict(lambda: [])

    for (emotion, channel, recording_name), blocks in picture_blocks.items():

        muscle = channel_nr_to_muscle(channel)
        person = recording_name_to_person(recording_name)
        block_length = int(len(blocks[0]) / fs)

        for block in blocks:

            for sample_start in range(0, block_length-sample_length, sample_step):

                sample = block[sample_start*fs:(sample_start+sample_length)*fs]
                data_samples[(emotion, muscle, person)].append(sample)

    return data_samples


def extract_features_for_all_data(data_samples):

    def extract_features(data_sample):
        return SignalFeatures(data_sample).__cfeatures__()

    final_data = []

    for (emotion, muscle, person), samples in data_samples.items():

        for i, sample in enumerate(samples):

            if i % 5 == 0:
                print("Extracting features...")

            data_row = extract_features(sample)
            data_row["Emotion"] = emotion
            data_row["Muscle"] = muscle
            data_row["Person"] = person
            data_row["Sample idx"] = i

            final_data.append(data_row)

    print("Features extracted.")
    return pd.DataFrame(final_data)


def save_data(data, file_name):
    data.to_pickle(f"{file_name}.pkl")


def plot_datasets(datasets):
    count = 0
    
    for key, value in datasets.items():
        ax = plt.gca()
        plt.title(key)
        value.plot(kind="line", x="Timestamp", y="EXG Channel 0", ax=ax, color="blue")
        value.plot(kind="line", x="Timestamp", y="EXG Channel 2", ax=ax, color="red")

        plt.show()
        
        count += 1


def run():
    #clean()
    datasets = read_data()
    datasets = clear_columns(datasets)
    datasets = apply_bp_filter(datasets)
    datasets = clear_edges(datasets)
    datasets = normalize_data(datasets)

    picture_blocks = extract_picture_blocks(datasets)

    sample_length = 2
    sample_step = 2
    data_samples = extract_data_samples(picture_blocks, sample_length, sample_step)

    final_data = extract_features_for_all_data(data_samples)

    save_data(final_data, f"data_sample_len{sample_length}_sample_step{sample_step}")

    print(final_data.head(1))
    # print_file_lengths(datasets)  # For debug to check if files are about the same length
    #plot_datasets(datasets)  # For debug to see plots of the filtered signals

    print("Collected data for %i src" % len(datasets))

if __name__ == "__main__":
    run()
