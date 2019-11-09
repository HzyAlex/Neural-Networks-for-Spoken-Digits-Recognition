import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import cfg
from MFCC import extract_mfcc_features_and_label


def evaluate_model_on_dataset(model, x, y):
    """ Evaluate model on data set (training/valid/test) set.

    Args:
    model: deep model
    Trained model
    x: numpy array
    Input data
    y: numpy array
    Input label

    Return:
    Evaluate loss and accuracy.
    """
    loss, acc = model.evaluate(x, y)
    return loss, acc


def confusion_matrix_on_dataset(model, x, y):
    """ Visualize the confusion matrix on data set (training/valid/test) set.

    Args:
    model: deep model
    Trained model
    x: numpy array
    Input data
    y: numpy array
    Input labels

    Return:
    Visualization of confusion matrix on dataset.
    """
    # Predict labels on dataset
    pred_y = model.predict(x)

    # Generate confusion matrix
    confuse_mat = confusion_matrix(np.argmax(y, axis=-1), np.argmax(pred_y, axis=-1), labels=np.arange(cfg.NUM_CLASSES))

    # Visualize
    fig, ax = plt.subplots(1, figsize=(10, 10))
    plt.imshow(confuse_mat)
    tick_locs = np.arange(cfg.NUM_CLASSES)
    ticks = ['{}'.format(i) for i in range(1, cfg.NUM_CLASSES + 1)]
    plt.xticks(tick_locs, ticks, fontsize='large')
    plt.yticks(tick_locs, ticks, fontsize='large')
    plt.ylabel("True number")
    plt.xlabel("Predicted number")
    plt.show()

    return confuse_mat


def evaluate_model_on_single_data(model, file_path, utterance_length=20):
    """ Evaluate trained model on single raw data.

    Args:
    model: deep model
    Trained model
    file_path: str
    Path of raw data
    utterance_length: int
    Length of data.
    Return: 
    Details of evaluate info.
    """
    # Preprocess and extract features from raw file
    mfcc_features, label = extract_mfcc_features_and_label(file_path, utterance_length)

    # Convert label from one-hot to normal format
    label = np.argmax(label, axis=-1)

    # Predict the label
    pred_label = np.argmax(model.predict(mfcc_features))

    # Show the evaluetion info
    print('File has been evaluated: {}'.format(file_path))
    print('Ground truth label: {}'.format(label))
    print('Predicted label: {}'.format(pred_label))

    return {"file_path": file_path, 'true_label': label, 'predicted_label': pred_label}