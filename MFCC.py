import os

import numpy as np
import librosa.display
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from config import cfg

def extract_mfcc(file_path, utterance_length=None):
    """ Extract MFCC features from single raw .wav file. 
    
    Args:
        file_path: str
            Path of raw .wav files.
        utterance_length: int
            Selected length of data to cut from or pad to.
    
    Return:
        MFCC features (flattened).
    """
    # Get raw .wav data and sampling rate from librosa's load function
    raw_w, sampling_rate = librosa.load(file_path, mono=True)
    
    # Obtain MFCC Features from raw data
    mfcc_features = librosa.feature.mfcc(raw_w, sampling_rate)
    
    # Cut or pad
    if utterance_length is not None:
        if mfcc_features.shape[1] > utterance_length:
            mfcc_features = mfcc_features[:, :utterance_length]
        else:
            mfcc_features = np.pad(mfcc_features, ((0, 0), (0, utterance_length - mfcc_features.shape[1])),
                                   mode='constant', constant_values=0)
            
    return mfcc_features.flatten()


def plot_raw_mfcc(file_path, utterance_length=None):
    """ Visualize the raw .wav file and MFCC features
        given path of files and utterance length.
        
    Args:
        file_path: str
            Path of raw .wav files.
        utterance_length: int
            Length of data to cut from or pad to.
    """
    # Get raw .wav data and sampling rate from librosa's load function
    raw_w, sampling_rate = librosa.load(file_path, mono=True)
    
    print('Shape of raw data: {}'.format(raw_w.shape))
    
    # Obtain MFCC Features from raw data
    mfcc_features = librosa.feature.mfcc(raw_w, sampling_rate)
    
    print('Selected utterance length: {}'.format(utterance_length))
    print('Shape of MFCC features before cutting or padding: {}'.format(mfcc_features.shape))
    
    # Cut or pad
    if utterance_length is not None:
        if mfcc_features.shape[1] > utterance_length:
            mfcc_features = mfcc_features[:, :utterance_length]
        else:
            mfcc_features = np.pad(mfcc_features, ((0, 0), (0, utterance_length - mfcc_features.shape[1])),
                                   mode='constant', constant_values=0)
    # visualize of raw data
    fig, ax = plt.subplots(1, figsize=(20, 6))
    ax.set_axis_off()
    plt.plot(raw_w)
    plt.title('Time series of raw data')
    plt.show()
    #visualize of extracted mfcc features
    fig, ax = plt.subplots(1, figsize=(20, 6))
    ax.set_axis_off()
#     plt.imshow(mfcc_features, interpolation='nearest')
    librosa.display.specshow(mfcc_features)
    plt.title('MFCC features')
    plt.colorbar()
    plt.show()
    
    print('Shape of MFCC features after flattening: {}'.format(mfcc_features.flatten().shape))
    
    
def shuffle_and_split(raw_data_dir, shuffle=True):
    """ Shuffle and randomly split all data into train/valid/test
    given splitting ratio from configurations.

    Args:
    raw_data_dir: str
    Directory of all raw data.
    shuffle: bool
    Whether to shuffle the order of files or not, default is True.
    """
    # load all file names and generate paths of all files 读取绝对路径
    all_raw_data_files = os.listdir(raw_data_dir)
    all_raw_data_paths = [os.path.join(raw_data_dir, raw_data_file) for raw_data_file in all_raw_data_files]
    # If we want to shuffle
    if shuffle:
        np.random.shuffle(all_raw_data_paths)

    # Calculate # of files on each dataset
    num_train_files = int(cfg.TRAIN_RATIO * len(all_raw_data_paths))
    num_valid_files = int(cfg.VALID_RATIO * len(all_raw_data_paths))

    # Split all data into train/valid/test set
    tr_raw_data_paths = all_raw_data_paths[:num_train_files]
    va_raw_data_paths = all_raw_data_paths[num_train_files:num_train_files + num_valid_files]
    te_raw_data_paths = all_raw_data_paths[num_train_files + num_valid_files:]
    print('Total # of raw files: {}'.format(len(all_raw_data_paths)))
    print('After splitting:')
    print('Training set contains {} files'.format(len(tr_raw_data_paths)))
    print('Validation set contains {} files'.format(len(va_raw_data_paths)))
    print('Test set contains {} files'.format(len(te_raw_data_paths)))

    return tr_raw_data_paths, va_raw_data_paths, te_raw_data_paths


def extract_mfcc_features_and_label(file_path, utterance_length=None):
    """ Extract MFCC features and label of single raw data. 

    Args:
    file_path: str
    Path of single raw file.
    utterance_length: int
    Length of data to cut from or pad to.

    Returns:
    mfcc_features: 1D numpy array, shape: (1, # of mfcc features, )
    label: 1D numpy array, shape: (1, # of unique labels, )
    """
    # Extract MFCC features from single file path given utterance length
    mfcc_features = extract_mfcc(file_path, utterance_length)

    print('MFCC features, before reshape: {}'.format(mfcc_features.shape))

    # Reshape the mfcc_features to meet the requirements
    mfcc_features = mfcc_features.reshape((1, -1))
    print('MFCC features, after reshape: {}'.format(mfcc_features.shape))

    # Extract labels from file names.
    basename = os.path.basename(file_path)

    # One-hot encoding for labels
    raw_label = int(basename.split('_')[0])
    label = np.eye(10)[raw_label]

    print('Shape of mfcc_feature: {}'.format(mfcc_features.shape))
    print('Shape of label: {}'.format(label.shape))

    return mfcc_features, label

def extract_all_mfcc_features_and_labels(file_paths, utterance_length=20):
    """ Extract MFCC features and labels of all raw data. 

    Args:
    file_paths: str
    List of paths of raw files.
    utterance_length: int
    Length of data to cut from or pad to.

    Returns:
    all_mfcc_features: 2D numpy array, shape: (# of samples, # of mfcc features)
    all_labels: 2D numpy array, shape: (# of samples, # of labels)
    """
    # Initialize the results
    all_mfcc_features = []
    all_labels = []

    # For loop through all files' paths
    for file_path in file_paths:

        # Extract MFCC features from single file path given utterance length
        mfcc = extract_mfcc(file_path, utterance_length)

        # Extract labels from file names.
        basename = os.path.basename(file_path)

        # One-hot encoding for labels
        raw_label = int(basename.split('_')[0])
        label = np.eye(10)[raw_label]

        all_mfcc_features.append(mfcc)
        all_labels.append(label)

    # Convert results from list of arrays to 2d arrays
    all_mfcc_features = np.asarray(all_mfcc_features)
    all_labels = np.asarray(all_labels)

    print('Shape of all_mfcc_features: {}'.format(all_mfcc_features.shape))
    print('Shape of all_labels: {}'.format(all_labels.shape))

    return all_mfcc_features, all_labels
