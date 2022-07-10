from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import time
import pandas as pd
import os
import numpy as np
import loadAndStoreData


def encodeLabels(labels):
    '''
    Transforms the labels into a one-hot encoded representation.
    '''
    encodedLabels = to_categorical(labels, dtype="int8")
    return encodedLabels


def labels_to_categorical(labels):
    '''
    Transforms the labels into their respective label number.
    '''
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels_categorical = le.transform(labels)
    return labels_categorical


def makePredictions(path: str, convnet: keras.Model, stepSize: int, windowSize):
    '''
    Traverses a folder that contains images for which predictions should be made.\n
    Creates a separate prediction CSV file for each image.

    @path - The path containing the images for which predictions should be created.
    '''
    for file in os.scandir(path):
        filepath = os.fsdecode(file)

        if(("annotated" in filepath) or not (filepath.endswith(".jpg") or filepath.endswith(".png"))):
            continue

        createPredictionsForImage(
            filepath=filepath, convnet=convnet, stepSize=stepSize, windowSize=windowSize)


def createPredictionsForImage(filepath: str, convnet: keras.Model, stepSize: int, windowSize):
    '''
    Creates the prediction CSV for one image.
    '''
    print("\nCreating predictions for file: ", filepath)
    create_predictions_start_time = time.time()
    imgArray = np.array(Image.open(filepath))

    patch_coordinates = []
    preprocessed_patches = []
    counter = 0
    patch_preprocessing_start_time = time.time()

    print("Starting sliding window to create patches of size: ",
          windowSize[0], "x", windowSize[1], ".")
    for(x, y, patch) in sliding_window(imageArray=imgArray, stepSize=stepSize, windowSize=windowSize):
        if counter > 0 and counter % 10000 == 0:
            print("Still processing, reached patch", counter)
            print("Execution time for the last 10.000 patches: ",
                  time.time()-patch_preprocessing_start_time, " seconds.")
            patch_preprocessing_start_time = time.time()
            print("Processing continues...")

        # Skip if the size of a patch doesn't match the specified windowSize
        if patch.shape[0] != windowSize[0] or patch.shape[1] != windowSize[1]:
            continue

        # Save coordinates which are needed for a prediction
        x_upper_left = x
        y_upper_left = y
        x_lower_right = x+windowSize[0]
        y_lower_right = y+windowSize[1]

        # Run the patch through the classification
        #preprocessed_patch = preprocess_input(patch)
        preprocessed_patches.append(patch)
        patch_coordinates.append(
            [y_upper_left, x_upper_left, y_lower_right, x_lower_right])
        counter += 1

    print("Finished preprocessing of the patches.")
    preprocessed_patches = np.array(preprocessed_patches)
    patch_coordinates = np.array(patch_coordinates)
    #print("Shape of preprocessed patches: ", preprocessed_patches.shape)
    #print("Shape of patch coordinates: ", patch_coordinates.shape, "\n")

    # Get all predictions
    print("Running patches through the Convnet...")
    prediction_start_time = time.time()
    predicted_labels_encoded = pd.DataFrame(convnet.predict(preprocessed_patches), columns=[
                                            "background", "ponds", "pools", "solar", "trampoline"])
    #print("Predicted labels encoded:", predicted_labels_encoded.head)
    predicted_labels = predicted_labels_encoded.idxmax(1)
    #print("Predicted labels decoded:", predicted_labels.head)

    print("Finished predictions, execution time: ",
          time.time()-prediction_start_time, " seconds.\n")

    highest_scores = predicted_labels_encoded[[
        "background", "ponds", "pools", "solar", "trampoline"]].max(axis=1)

    #print("Shape of patch_coordinates: ", patch_coordinates.shape)

    # Combining patch coordinates and predictions
    predictions_array = np.c_[highest_scores,
                              predicted_labels, patch_coordinates]

    #print("Shape of combined predictions array (unfiltered): ",predictions_array.shape)

    predictions_dataframe = pd.DataFrame(data=predictions_array, columns=[
                                         "score", "label", "y_upper_left", "x_upper_left", "y_lower_right", "x_lower_right"])
    # Filter all predictions that contain the label "background"
    predictions_dataframe = predictions_dataframe[predictions_dataframe.label != "background"]
    # print("Description of the predictions dataframe: ",predictions_dataframe.describe())

    predictions_dataframe.to_csv(os.path.split(filepath)[
                                 0]+"/"+os.path.split(filepath)[1].split(".")[0]+"_prediction.csv", sep=",", index=False)
    print("Saved predictions for file: ", filepath, "\n")
    print("Elapsed time: ", time.time() -
          create_predictions_start_time, " seconds.\n")


def sliding_window(imageArray, stepSize: int, windowSize=(256, 256)):
    for y in range(0, imageArray.shape[0], stepSize):
        for x in range(0, imageArray.shape[1], stepSize):
            # yield the current window
            yield (x, y, imageArray[y:y + windowSize[1], x:x + windowSize[0]])


def nonMaxSuppressBoundingBoxes(path: str, iou_threshold: float, score_threshold: float):
    '''
    Loads prediction csv files from the path and performs the non-max-suppression for each of them.\n
    This method works per-class, i.e. the suppression is performed for each object class independently.\n

    @path - The path in which the to-be-processed csv files are located.\n
    @iou_threshold - The percentage of allowed overlap for predictions of the same class.\n\t\t Must be a value between 0 and 1.\n
    @score_threshold - The minimum score a prediction must have to be considered values.\n\t\t Predictions with a score < score_threshold will be removed from the predictions\n.
    '''

    for file in os.scandir(path):
        filepath = os.fsdecode(file)

        # Skip files that are not csv files or that contain "suppressed" in their name
        if(not(filepath.endswith(".csv")) or ("suppressed" in filepath)):
            continue

        print("Creating suppressed csv for file: ", filepath, "...")
        # New empty dataframe for the results
        suppressed_predictions = pd.DataFrame(
            columns=["label", "y_upper_left", "x_upper_left", "y_lower_right", "x_lower_right"])

        # Get the original predictions from a csv file
        original_predictions = pd.read_csv(filepath, header=0)

        for pred_class in ["background", "pool", "pond", "solar", "trampoline"]:

            # Get labels, scores and coordinates for the class pred_class
            class_original_predictions = original_predictions.loc[
                original_predictions["label"] == pred_class]
            labels = class_original_predictions["label"]
            scores = class_original_predictions["score"]
            coordinates = class_original_predictions.iloc[:, 2:6].astype(int)

            # Run the nonmax suppression and gather the boxes and labels of the remaining predictions
            class_selected_boxes_indices = tf.image.non_max_suppression(
                boxes=coordinates, scores=scores, max_output_size=200, iou_threshold=iou_threshold, score_threshold=score_threshold)
            class_selected_boxes = tf.gather(
                coordinates, class_selected_boxes_indices).numpy()
            class_selected_labels = np.array(
                [x.numpy().decode() for x in tf.gather(labels, class_selected_boxes_indices)])
            class_predictions = pd.DataFrame(np.c_[class_selected_labels, class_selected_boxes], columns=[
                                             "label", "y_upper_left", "x_upper_left", "y_lower_right", "x_lower_right"])

            # Add the suppressed predictions of this class to the overall result
            suppressed_predictions = suppressed_predictions.append(
                class_predictions)

        # Save the suppressed predictions to a csv file
        new_filepath = os.path.splitext(filepath)[0]+"_suppressed.csv"
        suppressed_predictions.to_csv(new_filepath, sep=",", index=False)
        print("Success! Saved suppressed predictions to: ", new_filepath)
