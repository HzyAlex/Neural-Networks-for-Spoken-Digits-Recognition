import keras
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import Model, Sequential, load_model
import matplotlib.pyplot as plt
import seaborn as sns


def build_model(input_shape, hidden_sizes=[100], num_classes=10, print_summary=True):
    """ Build the deep model given input shape (data), hidden sizes and # of classes. 

    Args:
    input_shape: Tuple
    Example: (1400, 400)
    hidden_sizes: List of ints
    Example: [100] means model has only one hidden layer with 100 hidden units.
    [50, 50] means model has two hidden layers with 50 and 50 hidden units respectively.
    num_classes: Int
    # of classes (# of unique labels) to output.
    print_summary: bool

    Return:
    Keras deep model.
    """

    # Initialize the model
    model = Sequential(name='speech_recognition_model')

    # Build hidden layers  dense softmax
    for i, hidden_size in enumerate(hidden_sizes):
        if i == 0:
            hidden_layer = Dense(units=hidden_size, activation='sigmoid', name='hidden_layer_{}'.format(i),
            input_dim=input_shape[1])
        else:
            hidden_layer = Dense(units=hidden_size, activation='sigmoid', name='hidden_layer_{}'.format(i))
        model.add(hidden_layer)

    # Build output layer
    output_layer = Dense(units=num_classes, activation='softmax', name='output_layer')
    model.add(output_layer)

    # Print model summary
    if print_summary:
        print(model.summary())

    return model


def compile_model(model, lr=1e-3):
    """ Compile model in order to train it later. 

    Args:
    lr: float
    Learning rate.
    model_save_path: str
   
    Return:
    Compiled model.
    """
    # Initialize the optimizer 
    adam = Adam(lr=lr)

    # Compile the model
    model.compile(optimizer=adam,
    loss=categorical_crossentropy,
    metrics=[categorical_accuracy],)


    return model


def train_model(model, train_x, train_y, valid_x, valid_y, epochs=100, verbose=2,
    model_save_path='./saved_model.h5', patience=3):
    """ Train the model. 

    Args:
    model: deep model
    Compiled model
    train_x: numpy array
    Training set data
    train_y: numpy array
    Training set label
    valid_x: numpy array
    Validation set data
    valid_y: numpy array
    Validation set label
    epochs: int
    # of epochs to train the model
    verbose: int, choose from [0, 1, 2]
    Verbosity mode of model during the training process.
    model_save_path: str
    Path to save the best weights.
    patience: int
    # of epochs with no improvement on validation set, after which training will be stopped.

    Return:
    Trained model and training history.
    """
    # Initialize the model checkpoint, only save the best weights during the training process
    model_ckpt = ModelCheckpoint(filepath=model_save_path, save_best_only=True)

    # Initialize the early stopper, stop the training when validation loss starts increasing
    early_stop = EarlyStopping(patience=patience)

    # Training the model
    history = model.fit(x=train_x, y=train_y, epochs=epochs, verbose=verbose,
    callbacks=[model_ckpt, early_stop],
    validation_data=(valid_x, valid_y)
    )

    return model, history


def visualize_history(history):
    """ Visualize the training history. 

    Args:
    history: training history
    Returned by model training process.
    """
    # Get the loss and accuracy on training set
    train_loss = history.history['loss']
    train_acc = history.history['categorical_accuracy']

    # Get the loss and accuracy on training set
    valid_loss = history.history['val_loss']
    valid_acc = history.history['val_categorical_accuracy']

    # Get the # of training epochs
    num_epochs = len(train_loss)

    fig, ax = plt.subplots(1, figsize=(10, 6))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.legend(['train_loss', 'valid_loss'], fontsize='large')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    fig, ax = plt.subplots(1, figsize=(10, 6))
    plt.plot(train_acc, linestyle='--')
    plt.plot(valid_acc, linestyle='--')
    plt.legend(['train_acc', 'valid_acc'], fontsize='large')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()