# Importing
import xgboost
from joblib import dump, load
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import processing
import numpy as np
# print(ifxdaq.__version__)
import time
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
import matplotlib.pyplot as plt

config_file = "radar_configs/RadarIfxBGT60.json"

## Run this to understand the current radar settings better
import json

with open(config_file) as json_file:
    c = json.load(json_file)["device_config"]["fmcw_single_shape"]
    chirp_duration = c["num_samples_per_chirp"] / c['sample_rate_Hz']
    frame_duration = (chirp_duration + c['chirp_repetition_time_s']) * c['num_chirps_per_frame']
    print("With the current configuration, the radar will send out " + str(c['num_chirps_per_frame']) + \
          ' signals with varying frequency ("chirps") between ' + str(c['start_frequency_Hz'] / 1e9) + " GHz and " + \
          str(c['end_frequency_Hz'] / 1e9) + " GHz.")
    print('Each chirp will consist of ' + str(
        c["num_samples_per_chirp"]) + ' ADC measurements of the IF signal ("samples").')
    print('A chirp takes ' + str(chirp_duration * 1e6) + ' microseconds and the delay between the chirps is ' + str(
        c['chirp_repetition_time_s'] * 1e6) + ' microseconds.')
    print('With a total frame duration of ' + str(frame_duration * 1e3) + ' milliseconds and a delay of ' + str(
        c['frame_repetition_time_s'] * 1e3) + ' milliseconds between the frame we get a frame rate of ' + str(
        1 / (frame_duration + c['frame_repetition_time_s'])) + ' radar frames per second.')


# Funcitons

# Data preprocessing

def remove_center_line(arr):
    # find better solution for this
    for i in range(arr.shape[0]):
        curr_arr = arr[i]
        curr_arr[:][32] = 0
        arr[i] = curr_arr
    return arr


def preprocessing(data):
    data = data / 4095
    range_dopler_arr = np.abs(processing.processing_rangeDopplerData(data[:, 0, :, :]))
    range_dopler_arr = remove_center_line(range_dopler_arr)
    for i in range(1, 3):
        range_dopler_arr += np.abs(processing.processing_rangeDopplerData(data[:, i, :, :]))
        range_dopler_arr = remove_center_line(range_dopler_arr)

    range_dopler_arr = range_dopler_arr[:, :, :32]

    return range_dopler_arr


def removing_noise(n_frames, data):  # n_frames - number of frames for mean value
    mean_list = []
    clean_data = np.copy(data)

    first_mean = 0

    i = n_frames
    while (i < data.shape[0]):
        # for i in range(n_frames, data.shape[0]):
        new_session = False

        curr_mean = np.mean(data[i - n_frames:i], axis=0)
        if (i == n_frames): first_mean = curr_mean
        # mean_list.append(curr_mean)
        clean_data[i] -= curr_mean
        if (new_session):
            for j in range(i - n_frames, i):
                clean_data[j] -= curr_mean
        i += 1

    clean_data[:n_frames] -= first_mean

    return clean_data


def sum_feature(X):
    return np.sum(X, axis=(1, 2))


def max_feature(X):
    return np.max(X, axis=(1, 2))


def y_weights(X):
    y_mass = np.max(X, axis=2)
    y_mass = y_mass.T
    distance_arr = np.abs(32 - np.arange(y_mass.shape[0]))  # 64
    distance_arr = np.reshape(distance_arr, newshape=(distance_arr.shape[0], 1))

    return np.dot(y_mass.T, distance_arr)


def more_than(X):
    arr = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        one_frame = X[i]
        max_value = np.max(one_frame)
        n = len(np.where(one_frame >= 0.5 * max_value)[0])
        arr[i] = n

    return arr


def gradient(X):
    return np.gradient(X)


def weighted_value_at_the_middle(X):
    weighted_sum = 0
    for i in range(-31, 31):
        weighted_sum += np.sum(X[:, i, :] * np.exp(-abs(i)), axis=1)


    return weighted_sum


def removing_noise_2(data):  # n_frames - number of frames for mean value
    clean_data = np.copy(data)
    mean_pixels = np.mean(data, axis=0)
    for j in range(5):
        clean_data[j] -= mean_pixels

    return clean_data


# Collecting data

scaler = load('scaler_new.joblib')
model = load('model_new.joblib')

with RadarIfxAvian(config_file) as device:
    #i = input()
    lol = 5
    while (True):
        start_time = time.time()
        raw_data = []
        for i_frame, frame in enumerate(device):

            raw_data.append(np.squeeze(frame['radar'].data))
            if (i_frame == lol-1):

                data = np.asarray(raw_data)
                data_preproc = preprocessing(data)
                # data_clear = removing_noise(5, data_preproc)
                # del data
                # FEATURES
                sum_ft = sum_feature(data_preproc)  # sum of all pixels
                sum_ft = np.reshape(sum_ft, (sum_ft.shape[0]))

                max_ft = max_feature(data_preproc)  # max of all pixels
                max_ft = np.reshape(max_ft, (max_ft.shape[0]))

                y_mass_ft = y_weights(data_preproc)
                y_mass_ft = np.reshape(y_mass_ft, (y_mass_ft.shape[0]))

                n_more_than = more_than(data_preproc)
                n_more_than = np.reshape(n_more_than, (n_more_than.shape[0]))

                weighted_sum = weighted_value_at_the_middle(data_preproc)
                weighted_sum = np.reshape(weighted_sum, (weighted_sum.shape[0]))

                features = []
                features.append(sum_ft)
                df = pd.DataFrame()
                df["Sum"] = sum_ft
                # df["Label"] = Y
                df["Max"] = max_ft
                df["Y_mass"] = y_mass_ft
                df["More_than"] = n_more_than
                df["weighted_sum"] = weighted_sum
                pca = load('pca.joblib')
                converted_data = data_preproc.reshape(data_preproc.shape[0], -1)
                pca_model = pca.transform(converted_data)
                df = pd.concat([df, pd.DataFrame(pca_model)], axis=1)


                X = df
                X = scaler.transform(X)

                y_pred = model.predict(X)
                print(time.time() - start_time)
                print(y_pred)

                break
        #plt.imshow(data_preproc[-1])
        #plt.show()
        #i = input()