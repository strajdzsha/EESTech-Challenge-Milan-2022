# za sve row data vraca range dopler mape u odgovarajucem formatu

def remove_center_line(arr):
    #find better solution for this
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
