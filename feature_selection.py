import processing
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
import matplotlib.animation as animation
import pickle
import pandas as pd
import seaborn as sns


def remove_center_line(arr):
    #find better solution for this
    for i in range(arr.shape[0]):
        curr_arr = arr[i]
        curr_arr[:][32] = 0
        arr[i] = curr_arr
    return arr


def all_receivers(idx, amp_rdm):
    receiver_arr = np.zeros((64, 64))
    for j in range(n_receivers):
        curr_receiver = amp_rdm[idx][j]
        curr_receiver[:][32] = 0
        receiver_arr = receiver_arr + curr_receiver

    return receiver_arr


def preprocessing(data):
    data = data/4095
    range_dopler_arr = np.abs(processing.processing_rangeDopplerData(data[:, 0, :, :]))
    range_dopler_arr = remove_center_line(range_dopler_arr)
    for i in range(1, 3):
        range_dopler_arr += np.abs(processing.processing_rangeDopplerData(data[:, i, :, :]))
        range_dopler_arr = remove_center_line(range_dopler_arr)

    range_dopler_arr = range_dopler_arr[:, :, :32]

    return range_dopler_arr


def sum_feature(X):
    return np.sum(X, axis=(1, 2))


def max_feature(X):
    return np.max(X, axis=(1, 2))


n_receivers = 3
random_state = 420

X = np.load("train_data.npy")
Y = np.load("train_labels.npy")

X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.9, random_state=random_state)

X_rdm = preprocessing(X) # getting range doppler maps instead of raw data
img = X_rdm[100]
plot.imshow(img)
plot.show()

# FEATURES
sum_ft = sum_feature(X_rdm) # sum of all pixels
sum_ft = np.reshape(sum_ft, (sum_ft.shape[0]))

max_ft = max_feature(X_rdm) # max of all pixels
max_ft = np.reshape(max_ft, (max_ft.shape[0]))

features = []
features.append(sum_ft)
df = pd.DataFrame()
df["Sum"] = sum_ft
df["Label"] = Y
df["Max"] = max_ft

plot.figure(figsize=(12, 10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plot.cm.Reds)
plot.show()


