# Five-Project-Advanced-Deep-Learning-Low-Level-Modeling

This repository includes five Jupyter notebooks that demonstrate various deep learning models for different datasets and tasks.

## Project01.ipnyb :  
In this project, we use the Car-Price-Prediction dataset. The objective is to predict the price column based on other features. We build a model with three hidden layers, with each layer having 50 neurons. Further, we define a new loss function that calculated three different values (l1, l2, and l3) based on the difference between the actual value (y_true) and the predicted value (y_pred). We compare the performance of this loss function for 50 epochs with the mean_absolute_error and mean_squared_error losses, and plot it for the train and validation data.

## Project02.ipnyb : 
In Project02.ipnyb, the f-beta criterion (see the f-beta part of the link: https://en.wikipedia.org/wiki/F-score) is implemented in TensorFlow. A model with 3 hidden layers and 50 neurons in each layer is built to predict classes in the CIFAR10 dataset, and the value of this criterion is displayed for 50 epochs on the train and validation data.

## Project03.ipnyb : 
As you know, Dense layers in TensorFlow used the equation y = wx + b to calculate the output based on inputs , where w and b are learned during the process. However, the goal is to design a new custom layer that uses the equation y = vx^3 - wx + b to calculate the output based on the input. in this new layer  v, w, and b should be learned.
A model is built on the MNIST dataset to predict classes using two new custom layer. Only the parts of the MNIST dataset with outputs 0, 2, and 5 are used. The performance of this model is compared with the model designed with two Dense hidden layers for 50 epochs using train and validation data. The loss and accuracy are displayed for both models.

## Project04.ipnyb : 
In this project, a new convolutional layer has been created, which performs a convolution (3*3) and a convolution (5*5) on the input data, and then concate the results of these two together (tf.keras.layers.Concatenate()). The MNIST dataset has been modified like Project03 and a convolutional model has been built to predict the classes in this dataset. This model was built once with the usual two hidden convolutional layers in TensorFlow with the desired filter size and once with two new hidden convolutional layers created in this project. The performance of these two models has been evaluated for 100 epochs on train and validation data. Also, the fitting time of the two models has been compared.

## Project05.ipnyb : 
In this project, the existing model developed in Project01.ipynb is implemented with a custom training loop that runs for 100 epochs. During training, the loss and mean absolute error criteria are calculated and displayed for both the training and validation data at the beginning of training and every 10 epochs thereafter.
