import processing
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.animation as animation
from scipy import signal
fig, ax = plot.subplots()


def show_receiver_image(receiver_num, idx, sample_num):
    receiver_arr = range_doppler_map[idx*sample_num][receiver_num]
    plot.imshow(np.abs(receiver_arr))
    plot.show()


def max_all_frames(all_frames):
    return np.max(all_frames)


def all_receivers(idx, sample_num):
    receiver_arr = np.zeros((64, 64))
    for j in range(n_receivers):
        curr_receiver = amp_rdm[idx * sample_num][j]
        curr_receiver[:][32] = 0
        curr_receiver[63][63] = max_value/5
        receiver_arr = receiver_arr + curr_receiver

    return ax.imshow(receiver_arr, animated=True)


n_receivers = 3
raw_data = np.load("radar.npy")
print(raw_data.shape)
n_frames = raw_data.shape[0]
raw_data = raw_data / 4095


range_doppler_map = processing.processing_rangeDopplerData(raw_data)
amp_rdm = np.abs(range_doppler_map)
max_value = np.max(amp_rdm)
print(max_value)


ims = []
for i in range(n_frames):
    im = all_receivers(i, 1)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)


ani.save("movie.mp4")