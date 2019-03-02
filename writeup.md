# **Traffic Sign Recognition Writeup** 

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize, and visualize the data set
* Increase the size of the dataset with augmentation
* Design, train, and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/training_data_sample.png "Sample Training Data"
[image2]: ./output/data_histogram.png "Training Data Distribution"
[image3]: ./output/data_augmentation.png "Data Augmentation"
[image4]: ./output/augmented_data_histogram.png "Augmented Histogram"
[image5]: ./output/new_signs.png "New Traffic Signs"
[image6]: ./output/new_imgs_softmax.png "New Traffic Signs Softmax"


---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Numpy was used to assess the size of the training, validation, and test data sets:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

It is best to visualize the dataset before diving into model creation.  In this random sampling of training images, we see large variations in illumination across signs.  The distribution of images will also affect the performance of a model as it may underweight the likelihood of sparser classes.  As seen in the graph, the distribution is very uneven, suggesting the model might benefit from some data augmentation

![alt text][image1]

#### 3. Data Augmentation

Given the low number of training images for some of the classes, I decided to augment the training dataset such that every class had some minimum number of images.  I created randomizers that take a given image and alter them slightly in some dimension, including: rotation, Gaussian noise, salt and pepper noise, scale, brightness, and translation.  The allowable range of perturbation was tuned to still produce recognizable traffic signs of the correct class.  For example, rotating a sign 180° can give it an entirely different meaning.  Examples of each type of randomization are included below

![alt text][image3]

The new data distribution looks much better.  Each augmented image is produced by applying all the listed perturbations on a randomly selected image (from the corresponding class) in the original training data set.

![alt text][image4]

Data augmentation not only improves the distribution of training data but can also make the model more robust at dealing with new images by mimicking expected noise or differences in potential new images.  Augmenting data, even from from popular classes, is a good way to increase the overall size of the dataset and improve generalization of the model

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique.

My first step is to shuffle the training data to ensure that the order of the data does not influence the model

I also tested normalizing the image data by shifting them to zero mean and scaling pixel values to from -1 to 1.  This change should reduce the impact of relative differences between images (eg brightness, color) and reduces the size of the gradients during training time but my model performance degraded significantly when applied.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x38 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x38 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64   				|
| Fully connected		| Flattened input of 400, output of 120			|
| RELU					|												|
| Fully connected		| Input of 120, output of 84           			|
| RELU					|												|
| Fully connected		| Input of 120, output of 43           			|

 This is a deep CNN with two convolution layers followed by two fully connected layers that output the logit directly.  The loss is calculated as the sum of both the softmax cross entropy and L2 regularization


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The training specifications vary based on the host machine (eg local desktop vs cloud server) so you may want to tune some parameters (eg batch size) to suit your development environment.

The Adam optimizer was used to reduce the number of hyperparameters required relative to a standard SGD approach.  A learning rate of 0.001 was found to be effective.  Accuracy generaly increased with the number of epochs but performance gains decrease after ~30 epochs, depending on the complexity of the model and the use of dropout.  

Batch size had a large impact on model performance.  While there was some speedup with larger batches (256+), it was more than offset by the large resulting drop in accuracy.  Smaller batches produced higher accuracy, ceteris paribus, but ran significantly slower at the limit (<32).  For most of training, the batch size was set to 128, though the final models may have been run on smaller batches

Early models showed significant overfitting, with training accuracy exceeding validation accuracy by a large margin.  This was improved by adding L2 Regularization to disincentivize larger weights.  I also used dropout of various rates to minimize the number of 'dead' neurons and ensure that the entire model was being utilized though this required more epochs of training

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

**TODO**
My final model results were:

* Training set accuracy: ?
* Validation set accuracy: ?
* Test set accuracy: ?

Given the extremely large search space available for creating an optimal CNN traffic sign recognizer, I started with a simple approach and gradually developed my model from there.  The first design was modeled after the LeNet MNIST character reader we developed in Lecture 14, with slight modifications to account for the change in input data (eg 3 color channels vs greyscale).  It provided validation accuracies of 87.5% and 89.6% for 10 and 30 epochs, respectively

After reading the [Sermanet and LeCun paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), I decided to increase the number of filters used in the convolution layers to 38 and 64.  It took longer to train but the validation accuracy increased to 91.2% after 30 epochs.

Knowing that ReLUs, while potentially more resistant than sigmoid activation functions, can suffer from vanishing gradient issues, I devised to use leaky ReLUs instead.  It was only after I had implemented the change that I realized that tf.nn.leaky_relu() is supported on Tensorflow versions 1.4+, which was not available in my environment.  While further work in this area is interesting, I decided to focus on other aspects of the model instead

Interested in getting a better understanding of the model results, I started outputting training accuracy and quickly discovered that my models were overfitting the training data (eg  99.2% training accuracy for the 38/64 kernel model).  I first implemented dropouts between layers to minimize overfitting but found that my model's accuracy plummeted with a keep rate of 0.5 (validation accuracy <6%) unless I let it train for more epochs (20+).  I experimented with keep rates between 50%-90%.  L2 Regularization was also helpful in reducing overfitting, though, with both these techniques, there is also a risk of underfitting.

The number of filters in a CNN can have a large impact on its performance.  I tested a range of filter depths to optimize both model accuracy and training time.  Fewer filters generally trained much faster but performance started to degrade significantly below ~10/20 filters (first and second convolution, respectively).  On the other extreme, increasing the number of filters to 108/108, per the Sermanet paper, greatly increased training time with little improvement in accuracy.

I also experimented with different approaches of applying data augmentation.  At first, I used it just to increase the availability of rarer training images.  Then I tried augmenting every class with some fixed number of images.  Augmentation generally reduced overfitting but adding more than ~1,000 images per class made for very long training times

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

The first image has greater contrast than the ones in the training data set so, while it is easier to classify for a human, it may cause some confusion in the model.  Images 2 and 4 appear to have some glare that is likely the result of the watermark on the original image.  The third image is is relatively clean and clear but the last appears to have a small image gradient in the bottom left of the sign

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.


| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Vehicles over 3.5 metric tons prohibited      		|  Vehicles over 3.5 metric tons prohibited      		|
| Go straight or left     			| Go straight or left     			|
| Speed limit (70km/h)					|  Speed limit (70km/h)					|
| Roundabout mandatory	      		|  Roundabout mandatory	      		|
| Right-of-way at the next intersection			|  Right-of-way at the next intersection			|

The model correctly classifies all the images, which is aligns with the high test accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model is extremely confident in its classifications for each of the new images, as shown in the graphs below

![alt text][image6]