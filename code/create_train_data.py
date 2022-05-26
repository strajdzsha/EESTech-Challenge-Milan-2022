import numpy as np
import pickle
import os

### PACKAGES RAW DATA IN TRAIN AND LABEL NUMPY ARRAY
### DIMENSIONS OF OUTPUT ARRAYS X AND Y ARE (n_train, 3, 64, 128) and (n_train, 1) respectivly

print(os.listdir("labeled_data"))

X = np.empty((0, 3, 64, 128), float)
Y = np.empty((0, 1), int)
Z = np.empty((0, 1), int)

session = 0
for file_name in os.listdir("labeled_data"):
    with open("labeled_data\\" + file_name, "rb") as input_file:
        data = pickle.load(input_file)
        # X = np.concatenate((X, data[0]), axis=0)
        arr = np.ones(shape=(data[0].shape[0], 1))
        arr_y = arr * data[1]
        arr_z = arr * session
        Y = np.concatenate((Y, arr_y), axis=0)
        Z = np.concatenate((Z, arr_z), axis=0)

        session += 1

print(X.shape)
print(Y.shape)
print(Z.shape)
# with open("train_data.pkl", "wb") as output:
#     pickle.dump(X, output)
#
# with open("train_labels.pkl", "wb") as output:
#     pickle.dump(Y, output)

# with open("train_data.npy", "wb") as output:
#     np.save(output, X)
with open("train_labels.npy", "wb") as output:
    np.save(output, Y)
with open("session_labels.npy", "wb") as output:
    np.save(output, Z)

