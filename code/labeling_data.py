### THIS MODULE LABELS DATASET DEPENDING ON FILE NAMES

import os
import numpy as np
import pickle


def find_all_data(_root):
    paths = []
    i = 0
    for (root, dirs, files) in os.walk(_root, topdown=True):
        if "radar.npy" in files:
            tokens = root.split("\\")
            label = get_label(tokens[4])
            paths.append(tokens[4])
            i += 1

            add_label(root, label, i)
    return paths


def get_label(s):
    tokens = s.split("_")
    if tokens[1] == "one":
        return 1
    elif tokens[1] == "two":
        return 2
    elif tokens[1] == "three":
        return 3
    elif tokens[1] == "zero":
        return 0
    else:
        return -1


def add_label(root, label, idx):
    a = np.load(root + "\\" + "radar.npy")
    a_list = [a]
    a_list.append(label)
    with open("data" + str(idx) + ".pkl", "wb") as output:
        pickle.dump(a_list, output)


find_all_data("C:\\Users\\DELL\\all_recordings")

