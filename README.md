# CASE STUDY - Object Detection (Task II)

The case study consisted of an object detection task, where sample data was
provided for training a convolutional neural network that would be able to detect
trampolines, pools, ponds and solar roof applications within a satellite image.
The situation in the beginning of the case study was data-sparse, meaning there
was not a lot of data available for training a model. In particular, there only were
nine example images for ponds. Data augmentation would be necessary to create
a capable model for solving the task at hand and in addition to this, the group
decided on searching for additional labelled data online.


## Procedure
As a first step, we had to increase the amount of training data in order to train a
better and more sophisticated model. We used new, publicly available data-sets
of satellite images for the four objects as well as data augmentation techniques
on the provided training data. Since this is a multi-class classification scenario
with non-numerical classes, there is a need to encode the labels to numerical
values before fitting the data to the neural network. Afterwards, we can split the
data-set into a training and a validation set. The purpose of this split is only to
evaluate the initial classification performance of the neural network. For the
actual detection model that is used on large satellite images, we will use a model
that is fitted on both, the training and the validation data. To detect the images, a
sliding-window approach with a step size of [INSERT] pixels was used. Afterwards,
there is a need to apply a non-max suppression algorithm in order to combine
overlapping detections of objects to one single detection. Hereby, we used a
intersection-over-union threshold of zero and a score threshold of [INSERT].



## Convolutional Neural Network


## Initialization of working environment



## 


#
