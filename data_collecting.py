import pickle
import ifxdaq
import processing
import numpy as np
#print(ifxdaq.__version__)
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
import matplotlib.pyplot as plt

config_file = "radar_configs/RadarIfxBGT60.json"
number_of_frames = 5

## Run this to understand the current radar settings better
import json
with open(config_file) as json_file:
    c = json.load(json_file)["device_config"]["fmcw_single_shape"]
    chirp_duration = c["num_samples_per_chirp"]/c['sample_rate_Hz']
    frame_duration = (chirp_duration + c['chirp_repetition_time_s']) * c['num_chirps_per_frame']
    print("With the current configuration, the radar will send out " + str(c['num_chirps_per_frame']) + \
          ' signals with varying frequency ("chirps") between ' + str(c['start_frequency_Hz']/1e9) + " GHz and " + \
          str(c['end_frequency_Hz']/1e9) + " GHz.")
    print('Each chirp will consist of ' + str(c["num_samples_per_chirp"]) + ' ADC measurements of the IF signal ("samples").')
    print('A chirp takes ' + str(chirp_duration*1e6) + ' microseconds and the delay between the chirps is ' + str(c['chirp_repetition_time_s']*1e6) +' microseconds.')
    print('With a total frame duration of ' + str(frame_duration*1e3) + ' milliseconds and a delay of ' + str(c['frame_repetition_time_s']*1e3) + ' milliseconds between the frame we get a frame rate of ' + str(1/(frame_duration + c['frame_repetition_time_s'])) + ' radar frames per second.')

#%%

def show_img(data):
    fig, axs = plt.subplots(3, 5,figsize=(15,10),sharex=True, sharey=  True)
    for ax in axs.flat:
        ax.set_axis_off()
        
    for i in range(3):
        for j in range(5):
            ax = axs[i, j].imshow(np.angle(data)[j,i,:,:])
            axs[i, j].set_aspect('equal')
            #plt.savefig('testing/rd_plot_' + name)


    cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
    fig.colorbar(ax, cb_ax)

    #file_name ='dataset/' + name

    #plt.savefig(file_name, bbox_inches='tight')
    plt.show()

#%%
print('Pocetak: ')
i = int(input())
label = input()
while(label):
    
    raw_data = []
    with RadarIfxAvian(config_file) as device:                             
    
        for i_frame, frame in enumerate(device):
            
            raw_data.append(np.squeeze(frame['radar'].data/(4095.0)))
            if(i_frame == number_of_frames-1):
                data = np.asarray(raw_data)
                range_doppler_map = processing.processing_rangeDopplerData(data)
                break    
    
    file_raw_name ='dataset\\raw_' + str(i) + '.pkl'
    open_file = open(file_raw_name, 'wb')
    data_labled = [data, label]
    pickle.dump(data_labled, open_file)
    open_file.close()

    file_range_dopp_name ='dataset\\range_dopp_' + str(i) + '.pkl'
    open_file = open(file_range_dopp_name, 'wb')
    range_doppler_map_labled = [range_doppler_map, label]
    pickle.dump(range_doppler_map_labled, open_file)
    open_file.close()
    
    print(i)
    i += 1
    label = input()
#%%

open_file_0 = open('dataset\\range_dopp_-1.pkl', "rb")  #Sum
open_file_1 = open('dataset\\range_dopp_0.pkl', "rb")  #Strale
open_file_2 = open('dataset\\range_dopp_1.pkl', "rb")  #Strale Vojin
open_file_3 = open('dataset\\range_dopp_201.pkl', "rb") # Vojin

zero = pickle.load(open_file_0)
one = pickle.load(open_file_1)
two = pickle.load(open_file_2)
three = pickle.load(open_file_3)

open_file_0.close()
open_file_1.close()
open_file_2.close()
open_file_3.close()

open_file_4 = open('dataset\\range_dopp_200.pkl', "rb") # Vojin
four = pickle.load(open_file_4)

show_img(four[0])
show_img(three[0])

#show_img(zero[0])
#show_img(one[0])
#show_img(two[0])
#show_img(three[0])

#show_img(abs(one[0]/np.max(one[0])-two[0]/np.max(two[0])))  # Strale - Vojin+Strale
#show_img(abs(three[0]/np.max(three[0])-zero[0]/np.max(zero[0]))) # Vojin - Sum
#%%