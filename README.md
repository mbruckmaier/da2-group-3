# CASE STUDY - Object Detection (Task II)




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
