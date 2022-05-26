#import processing
#import numpy as np
#import matplotlib.pyplot as plot
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import roc_auc_score
#import matplotlib.animation as animation
#import pickle
#import pandas as pd
#import seaborn as sns
#import xgboost as xgb
#from sklearn.cluster import KMeans
import xgboost

import processing
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
import matplotlib.animation as animation
import pickle
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

def removing_noise(n_frames, data):  # n_frames - number of frames for mean value
    session_info = np.load("session_labels.npy")
    mean_list = []
    clean_data = np.copy(data)

    first_mean = 0

    i = n_frames
    while i < data.shape[0]:
        # for i in range(n_frames, data.shape[0]):
        new_session = False

        if session_info[i] != session_info[i - 1]:
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

    clean_data[:5] -= first_mean

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


n_receivers = 3
random_state = 420

X = np.load("train_data.npy")
Y = np.load("train_labels.npy")

#X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.85, random_state=random_state)

X_rdm = preprocessing(X) # getting range doppler maps instead of raw data
X_rdm = removing_noise(5, X_rdm)

img = X_rdm[1220]
print(Y[1220])
plot.imshow(img)
plot.show()

# FEATURES
sum_ft = sum_feature(X_rdm) # sum of all pixels
sum_ft = np.reshape(sum_ft, (sum_ft.shape[0]))

max_ft = max_feature(X_rdm) # max of all pixels
max_ft = np.reshape(max_ft, (max_ft.shape[0]))

y_mass_ft = y_weights(X_rdm)
y_mass_ft = np.reshape(y_mass_ft, (y_mass_ft.shape[0]))

n_more_than = more_than(X_rdm)
n_more_than = np.reshape(n_more_than, (n_more_than.shape[0]))

features = []
features.append(sum_ft)
df = pd.DataFrame()
df["Sum"] = sum_ft
df["Label"] = Y
df["Max"] = max_ft
df["Y_mass"] = y_mass_ft
df["More_than"] = n_more_than

plot.figure(figsize=(12, 10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plot.cm.Reds)
plot.show()


Feature = df[['Sum','Max', 'Y_mass', 'More_than']]
X = Feature.values
y = df['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
model_xgboost = xgboost.XGBClassifier(learning_rate=0.1,
                                      max_depth=5,
                                      n_estimators=5000,
                                      subsample=0.5,
                                      colsample_bytree=0.5,
                                      eval_metric="auc",
                                      verbosity=1)

eval_set = [(X_test, y_test)]
model_xgboost.fit(X_train, y_train, early_stopping_rounds=10, eval_set=eval_set, verbose=True)
y_train_pred = model_xgboost.predict_proba(X_train)[:,1]
y_test_pred = model_xgboost.predict_proba(X_test)[:, 1]
print("Train: {:.4f}". format(y_train, y_train_pred))
print("Test: {:.4f}". format(y_test, y_test_pred))