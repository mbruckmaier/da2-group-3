from PIL import Image
import numpy as np
import os
import pandas as pd


def loadImagesToArray(path: str):
    '''
    Loads all .jpg and .png files from the specified directory.\n
    Each image will be converted into an array of size (height x width x channels).\n
    The return numpy array is of dimensions (numberOfImages x height x width x channels).\n
    '''
    imagesArray = []

    counter = 0
    for file in os.scandir(path):
        filepath = os.fsdecode(file)
        if(filepath.endswith(".jpg") or filepath.endswith(".png")):
            imgArray = np.array(Image.open(filepath))
            imagesArray.append(imgArray)
            counter += 1
    return np.array(imagesArray)


def loadTrainingDataAndLabels(folders, subdirectories):
    '''
    Loads the training data as numpy arrays and creates the corresponding labels.\n
    For this to work, the images should be under the folder <path> in separate subdirectories, one for each class.\n
    The labels will be inferred from the names of the subdirectories. \n

    Returns the training data as a numpy array with the dimensions (number_of_images x height x width x channels).\n
    Returns the labels as a numpy array with the dimensions (number_of_images).
    '''

    final_data = []
    final_labels = []

    for folder in folders:

        training_data = []
        labels = []

        for directory in subdirectories:
            images_array = loadImagesToArray(os.path.join(folder, directory))
            training_data.extend(images_array)

            labels.extend(np.full(len(images_array), directory))

        training_data_array = np.array(training_data)
        labels_array = np.array(labels)
        print("Shape of training_data array: ", training_data_array.shape)
        print("Shape of labels array: ", labels_array.shape)
        final_data.extend(training_data_array)
        final_labels.extend(labels_array)

    final_training_data_array = np.array(final_data)
    print("Shape of final training_data: ", final_training_data_array.shape)
    final_labels_array = np.array(final_labels)
    print("Shape of final labels: ", final_labels_array.shape)

    return final_training_data_array, final_labels_array
