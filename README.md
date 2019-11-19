# Neural-Networks-for-Spoken-Digits-Recognition

This repositoy will demonstrate a project that is about a preliminary attempt for a utilization of different powerful tools, mainly neural network and machine learning, to create a simple-function speech recognition tool that can distinguish number 1 to 9 with a decent accuracy. 

## Abstract
Have you ever been captivated by the Intelligent robots like R2D2 or T-800 in science fiction movies? All of them have one thing in common—they are able to communicate with humans. Being able to talk is widely regarded as the most significant sign of intelligence, and to be able to do that, the first thing to do is to understand the word. And that’s when speech recognition comes out to play.

Making machines understand what people say has always been an ultimate goal for scientists. Speech recognition technology has a long history which can be traced back to 1950s. As it develops, and with the emergence of different types of tools, the error accuracy of speech recognition is now narrowed down to 4.9 percent.

## Techniques Used
Various techniques used in this project and they are all imported from different python libraries. As the original .wav audio files are difficult to process, Librosa library is used to convert them into MFCC features; Os library helps to create numpy arrays to encode each feature; Keras library enables the program to build a neural network and do machine learning; Matplotlib and Seaborn both help to visualize the process and results during the program.

# Procedures (click to see the whole [jupyter notebook](https://github.com/HzyAlex/Neural-Networks-for-Spoken-Digits-Recognition/blob/master/speech_recognition_all.ipynb))

## Part I - Reprocess And Extract MFCC Features From Raw Data
In order to properly train the model:

-   If Mfcc features (utterance length) are too long, we need to cut them. Otherwise, we need pad zeros to shorter ones.
    
-   Splitting all data into training/validation/test set given pre-defined ratio.
    
-   Extracting MFCC features and process labels of all raw data.

 Here we implement the following functions (see [MFCC.py](https://github.com/HzyAlex/Neural-Networks-for-Spoken-Digits-Recognition/blob/master/MFCC.py)):
* extract_mfcc: Extract MFCC features from single raw .wav file. 
* plot_raw_mfcc: Visualize the raw .wav file and MFCC features given path of files and utterance length.
* extract_mfcc_features_and_label: Extract MFCC features and label of single raw data.
* extract_all_mfcc_features_and_labels: Extract MFCC features and labels of all raw data. 
* shuffle_and_split: Shuffle and randomly split all data into train/valid/test given splitting ratio from configurations.

## Part II - Build, Compile And Train The Deep Model

Here we implement the following functions (see [Network.py](https://github.com/HzyAlex/Neural-Networks-for-Spoken-Digits-Recognition/blob/master/Network.py)):

-   build_model: Build the deep model given input shape (data), hidden sizes and # of classes
-   compile_model: Compile model in order to train it later
-   train_model: Train the model
-   visualize_history: Visualize the training history

## Part III - Evaluate The Trained Model

Here we implement the following functions (see [Evaluation.py](https://github.com/HzyAlex/Neural-Networks-for-Spoken-Digits-Recognition/blob/master/Evaluation.py)):


-   evaluate_model_on_dataset: Evaluate model on data set (training/valid/test) set.
-   evaluate_model_on_single_data: Evaluate model on single file.
-   confusion_matrix_on_dataset: Generate and visualize the confusion matrix on test set.

# END
For more details, view report.doc.







