import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from scipy.io import loadmat


def create_autoencoder(data, input_size=174, latent_dim=3, dimensions=[32, 16], epochs=100, batch_size=32, loss='mean_squared_error', optimizer='adam'):

    # Define the autoencoder model
    input_data = Input(shape=(input_size,))
    # Encoder
    encoded = Dense(dimensions[0], activation='relu')(input_data)
    encoded = Dense(dimensions[1], activation='relu')(encoded)
    encoded = Dense(latent_dim)(encoded)  # Latent space representation

    # Define the encoder model
    encoder = Model(input_data, encoded)

    # Decoders
    decoded = Dense(dimensions[1], activation='relu')(encoded)
    decoded = Dense(dimensions[0], activation='relu')(decoded)
    decoded = Dense(input_size)(decoded)

    # Combine encoder and decoder to create autoencoder model
    autoencoder = Model(input_data, decoded)

    # Compile the model
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    #fit the model
    autoencoder.fit(data, data, epochs=50, batch_size=32, verbose=1)

    return autoencoder, encoder

if __name__=="__main__":
    # Define the input size
    input_size = 174

    # Generate random data for demonstration
    input_data = np.random.rand(1000, input_size)

    # Create an autoencoder model
    autoencoder, encoder = create_autoencoder(data=input_data, input_size=input_size, latent_dim=3)
    
    # predict the autoencoder output from test data
    decoded_data = autoencoder.predict(input_data)
    print(decoded_data.shape)

    # Viusalize the latent space
    encoded_data = encoder.predict(input_data)
    print(encoded_data.shape)

def get_spM(data, t):
    #kick the unsorted spikes u1
    sorted_spikes = data[:,1:]

    #get the spike matrix
    spM = []
    n_channels, n_units = sorted_spikes.shape
    for channel in range(n_channels):
        for unit in range(n_units):
            spike_times = sorted_spikes[channel, unit].flatten()
            if len(spike_times) > 0:
                spM.append(np.histogram(spike_times, bins=t)[0])

    spM = np.array(spM)
    return spM

def get_clean_data(file):
    """
    load the data from the file and return the spike matrix, event matrix, hand matrix, cursor matrix, time vector and target matrix

    Parameters
    ----------
    file : str
        the name of the file to load
    
    Returns
    -------
    dict
        a dictionary containing the spike matrix, event matrix, hand matrix, cursor matrix, time vector and target matrix
        
    t is the timestamp corresponding to each sample of the cursor_pos, finger_pos, and target_pos, seconds.
    """
    data = loadmat('indy_20160407_02_py.mat')

    #A. Get the instantaneous firing rate: from spike-times obtain an instantaneous firing rate (continuous time series)
    #kick the unsorted spikes u1
    sorted_spikes = data['spikes'][:,1:]
    time_bins = data['t'].flatten()
    spM = []
    n_channels, n_units = sorted_spikes.shape
    for channel in range(n_channels):
        for unit in range(n_units):
            spike_times = sorted_spikes[channel, unit].flatten()
            if len(spike_times) > 0:
                spM.append(np.histogram(spike_times, bins=time_bins)[0])

    spM = np.array(spM)

    # B. build an event matrix evM (PxT) of [0 1] marking the presence/absence of a certain stimulus
    evM = spM > 0

    # C. build a hand position matrix handM (3xT) with the coordinates of the hand
    handM = data['finger_pos'].T

    dict_data = {'spM': spM, 
                 'evM': evM, 
                 'handM': data['finger_pos'].T, 
                 'cursorM': data['cursor_pos'].T,
                 't': data['t'].flatten(),
                 'targetM': data['target_pos'].T,
                 } 
    return dict_data

