# CASE STUDY - Object Detection (Task II)

The case study consisted of an object detection task, where sample data was
provided for training a convolutional neural network that would be able to detect
trampolines, pools, ponds and solar roof applications within a satellite image.
The situation in the beginning of the case study was data-sparse, meaning there
was not a lot of data available for training a model. In particular, there only were
nine example images for ponds. Data augmentation would be necessary to create
a capable model for solving the task at hand and in addition to this, the group
decided on searching for additional labelled data online.


</br>


## Initialization of working environment


In order to run the object detection algorithm, you will need to set up a Google Account. The data that is necesssary for for the model training needs to be shared with your personal Google colab account. Therefore, please send your Google mail adress with us, if you would like to have access to the data. When your account has the necessary rights for the file, you will find the shared folder, as usual, in the shared files part of Google Drive. In order to load the necessary data, you will have to load the shared folder into the personal google MyDrive. 

The first part of the notebook includes the mounting of the Drive into the runtime environment. Further, you will need to intall the opencv module into the google colab environment.

</br>

## Convolutional Neural Network
The neural network would be the essential part of the entire project. For creating
a neural network, the Python package Tensorflow, kindly provided by Google, was
used to create Keras-Models. At first, we tried to build a model from scratch, but
after trying out multiple different combinations of layers and depth, we found out
that the resulting f1 score on the given validation images was never higher than
5%. Therefore, we focused on pretrained CNN image classifying models and
tested four different models: VGG-16, ResNet50, InceptionV3 and EfficientNet.In
each model, we used pretrained weights from the ImageNet dataset. We removed
the last layer of the models and replaced them with a simple fully connected
output layer with five nodes for each label and a softmax activation function to
output final predictions for probabilities for each class.

![vgg16-1-e1542731207177](https://user-images.githubusercontent.com/44417612/178339854-60d72da4-e7ab-4b0d-91d2-23603a5be863.png)


</br>

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

![da2process](https://user-images.githubusercontent.com/44417612/178347446-8c1737b1-d76c-468a-b40b-b98cd621c19b.png)


</br>

## Contributors
- Julian Granitza
- Jan-Ole van Wüllen
- Mathias Bruckmaier
- Vuk Stojkovic
- Alexander Wesenberg

</br>

## Individual Contributions
- Julian Granitza: CNN Setup and Assessment, Performance Tests, Poster Content, Additional Data Aquisition
- Jan-Ole van Wüllen: CNN Setup and Assessment, Code Integration, Performance Tests, README.md, Poster Content
- Mathias Bruckmaier: CNN Setup and Assessment, Non-max Suppression, Sliding-Window Forecast, Additional Data Aquisition,Label Classification of additional solar panels, Code Integration, Performance Tests, README.md, Poster Content
- Vuk Stojkovic: Label Classification of additional ponds and pools, Setup of Latex Template
- Alexander Wesenberg: Google Colab Setup, Data Augmentation, Code Integration, Performance Tests, README.md, Poster Content
