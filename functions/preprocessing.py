# za sve row data vraca range dopler mape u odgovarajucem formatu

def preprocessing(data):
    data = data/4095
    range_dopler_arr = np.abs(processing.processing_rangeDopplerData(data[:, 0, :, :]))
    for i in range(1, 3):
        range_dopler_arr += np.abs(processing.processing_rangeDopplerData(data[:, i, :, :]))
    
    return range_dopler_arr
