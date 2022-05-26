import processing
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.animation as animation
import pickle
import pandas as pd
import seaborn as sns


def removing_noise(n_frames, data):  # n_frames - number of frames for mean value
    session_info = np.load("session_labels.npy")
    mean_list = []
    clean_data = np.copy(data)

    first_mean = 0

    i = n_frames
    while (i < data.shape[0]):
        # for i in range(n_frames, data.shape[0]):
        new_session = False

        if (session_info[i] != session_info[i - 1]):
            new_session = True
            i += n_frames

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
    data = data / 4095
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


def y_weights(X):
    y_mass = np.max(X, axis=2)
    y_mass = y_mass.T
    distance_arr = np.abs(32 - np.arange(y_mass.shape[0])) # 64
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
    for i in range(-31,31):
        weighted_sum += np.sum(X[:, i, :] * np.exp(-abs(i)), axis=1)

    print(weighted_sum.shape)

    return weighted_sum


n_receivers = 3
random_state = 420

X = np.load("train_data.npy")
Y = np.load("train_labels.npy")

#X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.01, random_state=random_state)
X, Y = shuffle(X, Y, random_state= random_state)

X_rdm = preprocessing(X) # getting range doppler maps instead of raw data
X_rdm = removing_noise(5, X_rdm)


# FEATURES
sum_ft = sum_feature(X_rdm) # sum of all pixels
sum_ft = np.reshape(sum_ft, (sum_ft.shape[0]))

max_ft = max_feature(X_rdm) # max of all pixels
max_ft = np.reshape(max_ft, (max_ft.shape[0]))

y_mass_ft = y_weights(X_rdm)
y_mass_ft = np.reshape(y_mass_ft, (y_mass_ft.shape[0]))

n_more_than = more_than(X_rdm)
n_more_than = np.reshape(n_more_than, (n_more_than.shape[0]))

weighted_sum = weighted_value_at_the_middle(X_rdm)
weighted_sum = np.reshape(weighted_sum, (weighted_sum.shape[0]))

# PCA features selectoin

from sklearn.decomposition import PCA

n_pca = 8
pca = PCA(n_pca)
converted_data = X_rdm.reshape(X_rdm.shape[0], -1)
pca_model = pca.fit_transform(converted_data)
print(pca_model.shape)
print(sum(pca.explained_variance_ratio_))

features = []
features.append(sum_ft)
df = pd.DataFrame()
df["Sum"] = sum_ft
df["Label"] = Y
df["Max"] = max_ft
df["Y_mass"] = y_mass_ft
df["More_than"] = n_more_than
df["weighted_sum"] = weighted_sum
df = pd.concat([df, pd.DataFrame(pca_model)], axis=1)

print(df.shape)
df.to_csv("data.csv")

# Plotting correlations

plot.figure(figsize=(24, 20))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plot.cm.Reds)
plot.show()