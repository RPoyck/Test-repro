# Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# Overview
---
In this project, I have used what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model was trained, it was tested on 5 random images of German traffic signs found on the web.

The included Ipython notebook is based heavily on the starter code and instructions in the example [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

To meet specifications, the project submission consist of three files: 
* The [Ipython notebook](Traffic_Sign_Classifier-Rob_Poyck.ipynb) with the code.
* The code exported as an [html file](Traffic_Sign_Classifier-Rob_Poyck.html).
* A write-up report either as a markdown file which you are now reading. 

# The Project
---
The goals / steps of this project are the following:
 * Load the data set
 * Explore, summarize and visualize the data set
 * Design, train and test a model architecture
 * Use the model to make predictions on new images
 * Analyse the softmax probabilities of the new images
 * Summarize the results with a written report

# Loading and evaluating the datasets
The data was provided was already pickled and divided in training, validation and testing datasets, which are simply opened and divided in their features and labels. 

## Data size
Evaluation of this data shows:
 * Number of training examples = 34799
 * Number of validation examples = 4410
 * Number of testing examples = 12630
Which means there are enough examples for the training with sensible proportionality between the training and validation sets. The test dataset seems to have quite a lot of examples compared to the training dataset, which will however provide a good view of the workings of the final network. 

## Data form
What can also be seen from the datasets is:
 * Image data shape = 32x32
 * Number of classes of traffic signs = 43
Which is how it is described and expected. The maximum and minimum values of the colour indices of all images have also been evaluated once, the code of which is commented out in the jupyter notebook. As expected the minimum value is 0 and the maximum value is 255, this will thus be the input for later normalisation.

## Data content and quality <a name="Data_con_&_qual"></a>
As for the physical content of the datasets, a random list of 10 signs with their label and label meaning is shown as output during every run. This has given the following insights:
 * The low resolution of the images results in a satisfactory identifiablity for the shape of the shield, but not of the details contained in it. This means that classes of signs (warnings, prohibitions, obligations, etc.) are distinguishable, but the contents (actual limit speed, or type of warning e.g. wild animal, bumps or children crossing) can sometimes not be clear at all. Such as in:  
![image 1](/Writeup-images/Image_2-Slippery-road.png "Slippery road") and ![image 2](/Writeup-images/Image_3-Slippery-road.png "Slippery road").  
 * Some of the images have such a low resolution or lack of contrast that even I cannot distiguish what sign it is. Since these signs are still classified in the network somehow, this might effect the quality of the network. Such as in:  
![image 3](/Writeup-images/Image_1-Keep_right.png "Keep right") and ![image 4](/Writeup-images/Image_4-No_passing_for_vehicles_over_3.5_metric_tons.png "No passing for vehicles over 3.5 metric tons").  
 * The labels I have inspected always match the signs on the images as far as the signs are distinguishable legible.

## Data diversity
Additionally there is a great spread in the number of examples from each traffic sign class in the training dataset, a full listing of which is provided in the jupyter notebook. There are only 180 examples of the "End of no passing by vehicles over 3.5 metric tons" and 2010 examples of the	"Speed limit (20km/h)" sign. This might lead to the neural network having a bias to a certain solution, or at least to a decreased security of classifying the signs which have less examples.

# Model architecture and tests

## Pre-processing the data and defining hyperparameters
The number of **epochs** is chosen to be **15**, because this number in multiple runs showed the highest stability of reaching a validation accuracy of at least 93% while not going down at the end due to overfitting. In essence this can be seen as a hard programmed *Early termination*.  
The default **batch size** of **128** has proven to be fine for my computational power.  
A boolean hyperparameter for preprocessing the data to convert it to **grayscale** has been included to test the effect of this grayscale conversion. During the tests preformed the effect of grayscaling was quite minimal. This could be due to the negation of the added simplicity because the colour information also adds an extra identification property, especially for traffic signs.  
The default **learning rate** of **0.001** has been proven to be a stable value in order to converge quickly enough and most of all in a stable manner without significant overshoot.  
In order to speed up the learning process, weight decay and estimation and to avoid getting stuck in local minima the image data is **normalised** before being used.

## Model architecture
This is has mainly been copied from the already available LeNet example, because this is good at classifying letters and numbers, which are also an important element in traffic signs. It consists of:
 * Convolution layer 1. 
     * Input shape 32x32x3 or 32x32x1 depending on use of grayscaling. 
     * Output shape 28x28x6. 
     * Classifies simple shapes.
 * Activation 1. 
     * Relu. which is one of the simplest non-linear functions, which is easy and light while still being able to represent non linear features.
 * Pooling layer 1. 
     * Output shape 14x14x6. 
     * This decreases the output size which reduces the number of parameters in the consequtive layers, for faster processing. Because multiple pixels are combined the network will be less sensitive for pattern shifts.
 * Convolution layer 2. 
     * Output shape 10x10x16. 
     * Classifying more complex composite shapes.
 * Activation 2. 
     * Relu, same reason as activation 1.
 * Pooling layer 2. 
     * Output shape 5x5x16. 
     * Same reason as pooling 1.
 * Flatten layer. 
     * Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. 
     * 400 outputs.
 * Fully connected layer 1. 
     * 120 outputs.
 * Activation 3. 
     * Relu, same reason as activation 1.
 * Dropout 1. 
     * Chosen to reduce overfitting initially observed by the higher increase of the training set accuracy compared to the validation accuracy increase. It does this by random setting part of the output values to 0, training the network to handle uncertainty in the layer interface and forcing it to introduce redundancy.
 * Fully connected layer 2. 
     * 84 outputs.
 * Activation 4. 
     * Relu, same reason as activation 1.
 * Dropout 2. 
     * Same reason as dropout 1.
 * Fully connected layer 3. 
     * 43 outputs which are the logits predicting the class of the sign in the image.

## Training pipeline <a name="trainingPipe"></a>
Any supplied batch of features is passed through the neural network, after which the given predictions/logits are assessed for correctness with the labels of the same batch by calculating the cross-entropy. Stochastic gradient descent is used to reduce the error of the predictions by using the Tensorflow Adam (ADAptive Moment estimation) optimiser, where the running averages of both the gradients and the second moments of the gradients are used to try and reduce the models' cross entropy by addapting the weights accodingly.

## Model evaluation <a name="modelEval"></a>
During and after training the performance is assessed by the model evaluation function. This divides the evaluation set in batches. Each batch of features is passed through the neural network, after which the given predictions/logits are assessed for correctness with the labels of the same batch. Finally the comulative procentual correctness or accuracy is returned.

## Model training and validation
The entire training dataset is worked through for every one of the epochs defined above. Every epoch: 
 * The training images are **shuffled** in order to avoid any sequential relationship in the dataset to lead to learning imbalances.
 * The evaluation set is divided up into batches, for each of which:
     * The batch is passed through the [training pipeline](#trainingPipe) above.
 * The addapted model is used the [model evaluation](#modelEval) above with the:
     * Training dataset to determine the training accuracy. Because this is biased since the network was trained on the same physical features, the network is also tested with the:
     * Validation dataset to determine the validation accuracy.
After the training the trained network is saved for later use.

## Testing
Using the [model evaluation](#modelEval) above with the totally independent testing dataset to determine the testing accuracy. This independence is necessary because the training data was already used for training the network. And since the validation data was also used every epoch it could have biased the network to the correct values for this data set.

# Testing the trained network on random new images
In order to assess the versitility and flexibility of the system 5 random German traffic sign images of 5 of the defined classes were taken from the internet and passed through the network.  

## Network output
For each of the traffic signs the top 5 possible answers, ranked by probability as perceived by the network, are displayed with their meaning and probability assumption.  
![image_r_1](/Writeup-images/11.resized.jpg "Right-of-way at the next intersection")  
Actual sign:  11 (Right-of-way at the next intersection)  
Prediction 0: 11 (Right-of-way at the next intersection) with a certainty of: 100%  
Prediction 1: 30 (Beware of ice/snow) with a certainty of: 0%  
Prediction 2: 21 (Double curve) with a certainty of: 0%  
Prediction 3: 40 (Roundabout mandatory) with a certainty of: 0%  
Prediction 4: 27 (Pedestrians) with a certainty of: 0%  
<br>

![image_r_2](/Writeup-images/18.resized.jpg "General caution")  
Actual sign:  18 (General caution)  
Prediction 0: 18 (General caution) with a certainty of: 86%  
Prediction 1: 26 (Traffic signals) with a certainty of: 13%  
Prediction 2: 27 (Pedestrians) with a certainty of: 0%  
Prediction 3: 24 (Road narrows on the right) with a certainty of: 0%  
Prediction 4: 11 (Right-of-way at the next intersection) with a certainty of: 0%  
<br>

![image_r_3](/Writeup-images/21.resized.jpg "Double curve")  
Actual sign:  21 (Double curve)  
Prediction 0: 28 (Children crossing) with a certainty of: 96%  
Prediction 1: 20 (Dangerous curve to the right) with a certainty of: 3%  
Prediction 2: 11 (Right-of-way at the next intersection) with a certainty of: 0%  
Prediction 3: 23 (Slippery road) with a certainty of: 0%  
Prediction 4: 30 (Beware of ice/snow) with a certainty of: 0%  
<br>

![image_r_4](/Writeup-images/25.resized.jpg "Road work")  
Actual sign:  25 (Road work)  
Prediction 0: 25 (Road work) with a certainty of: 100%  
Prediction 1: 20 (Dangerous curve to the right) with a certainty of: 0%  
Prediction 2: 22 (Bumpy road) with a certainty of: 0%  
Prediction 3: 31 (Wild animals crossing) with a certainty of: 0%  
Prediction 4: 10 (No passing for vehicles over 3.5 metric tons) with a certainty of: 0%  
<br>

![image_r_5](/Writeup-images/31.resized.jpg "Wild animals crossing")  
Actual sign:  31 (Wild animals crossing)  
Prediction 0: 18 (General caution) with a certainty of: 88%  
Prediction 1: 26 (Traffic signals) with a certainty of: 9%  
Prediction 2: 0 (Speed limit (20km/h)) with a certainty of: 1%  
Prediction 3: 38 (Keep right) with a certainty of: 0%  
Prediction 4: 4 (Speed limit (70km/h)) with a certainty of: 0%  
<br>

## Assessing the results
The conclusion from the output is that the neural network has successfully identified 100% of the traffic sign **shapes**, only the content within this shape was misidentified for 40% of the signs. In both cases of misclassification the neural network also indicated that it was less than 100% sure of this prediction, while two of the correct identification had an assumed certainty of 100%. This at least shows that the network could make the user aware of the fact that it was not completely sure of the prediction.  

Possible reasons for the misclassification:
 * The network has not been trained such that it achieves an accuracy of close to 100%, thus it is not to be expected that this score becomes better with even less related pictures.
 * The misidentified signs all had a less than average amount of examples in the test dataset.
 * The low resolution and contrast of (some of) the test images as mentioned in the [Data content and quality](#Data_con_&_qual) section, will lead to difficulties of classifying details if the are hard to impossible to distinguish.
 * There are actually two different "double curve" signs, one which goes left first and one which goes right first. These should have been split up in the classification. Possibly the test dataset also contains (mainly) examples of going left first in stead of my image where the first curve is to the right. The latter could of course be tested by mirroring my test image.
 * The number of the test images is far to small to indicate any statistical accuracy of the network.

# Improvement propositions
The following possible improvements can be tried to further optimise the neural network:
 * Expanding the training dataset
     * An easy way to (virtually) expand the dataset would be to do multiple augmentations on the training images, which can result in a more robust network which is less dependent of exact sign shapes, orientations and positions.
     * An even faster way would be to move part of the test dataset to the training dataset, since it is realatively large to begin with. This would however increase the uncertainty of the correctness of the test results.
 * Using higher resolution images for training.
 * Discarding training images which have a contrast below a certain value.
 * Further tuning the hyperparameters. In particular the output sizes of the convolutional layers to pick up the right sequence of shapes from the images, possibly by using the visualisation technique shown in [Zeiler and Fergus](http://www.matthewzeiler.com/wp-content/uploads/2017/07/eccv2014.pdf).
 * The classification of individual classes can be assessed, i.e. looking at the percentage of correct estimation over the entire test data-set per class. This will provide more insight if certain traffic sign have a perticularly low classification accuracy, after which a more focussed improvement plan can be defined.


# Dependencies
For this lab the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) was used.
