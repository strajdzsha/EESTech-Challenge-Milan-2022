# za sve dopler range mape vraca podatke sa otklonjenim sumom u odgovarajucem formatu

# Removing noise
                  
def removing_noise(n_frames, data):  # n_frames - number of frames for mean value
    mean_list = []
    clean_data = np.copy(data)
    
    first_mean = 0
    
    i = n_frames
    while(i<data.shape[0]):
    #for i in range(n_frames, data.shape[0]):
        new_session = False
        
        if(session_info[i] != session_info[i-1]):
            new_session = True
            i += n_frames
            
        curr_mean = np.mean(data[i-n_frames:i], axis=0)
        if (i==n_frames): first_mean = curr_mean
        #mean_list.append(curr_mean)
        clean_data[i] -= curr_mean
        if (new_session):
            for j in range(i-n_frames, i):
                clean_data[j] -= curr_mean
        i += 1
    
    clean_data[:5] -= first_mean
    
    return clean_data