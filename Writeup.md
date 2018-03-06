# Project #2: **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project were the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/sample_image.png "Sample Training Image"
[image2]: ./writeup_images/training_histogram.png "Training Data Histogram"
[image3]: ./writeup_images/validation_histogram.png "Validation Data Histogram"
[image4]: ./writeup_images/test_histogram.png "Test Data Histogram"
[image5]: ./random_test_images/30kmph_label1.jpg "30kmph - Label 1"
[image6]: ./random_test_images/ChildrenCrossing_label28.jpg "Children Crossing - Label 28"
[image7]: ./random_test_images/EndAllLimits_label32.jpg "End All Limits - Label 32"
[image8]: ./random_test_images/NoEntry_label17.jpg "No Entry - Label 17"
[image9]: ./random_test_images/Yield_label13.jpg "Yield - Label 13"
[image10]: ./writeup_images/LeNetArch.png "LeNet Architecture"

## Rubric Points
### Here I have considered the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and described how I addressed each point in my implementation.  

---
### Writeup

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! This is the Writeup, and here is a link to my [project code](https://github.com/antriksh1/CarND-Traffic-Sign-Classifier-P2-AS/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: _34799 images._
* The size of the validation set is: _4410 images._
* The size of test set is: _12630 images._
* The shape of a traffic sign image is: _32 pixels x 32 pixels x 3 channels (RGB)_
* The number of unique classes/labels in the data set is: _43 labels._

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

First, I just want to take a look at a single random image, from the training data-set:

![Training Data-Set Sample Image][image1]

Next, lets plot the distribution of labels in the Training Data-Set, so we know which labels occur more, 
and should be correctly classified:
![Training Data Histogram][image2]

Now, lets also plot the distribution of labels in the Validation Data-Set, to determine how closely it mirrors the training data-set. 
![Validation Data Histogram][image3]

Now, we can see both histograms are similar, however, they are skewed toward the left, which means we have more samples of labels #0-19, than of labels #20-42.
This means that once we train the model, it should be much accurate at predicting labels with higher number of samples available (e.g. #1, #2, #10 etc.), than the ones with less samples (e.g. #21, #22, #41, #42)

Finally, lets also plot the distribution of labels in the Test Data-Set, again, to determine how closely it mirrors the training data-set. 
![Test Data Histogram][image4]

Since this also mirrors the Training data-set histogram and the Validation data-set histogram, this means that the test-set evaluation should be fairly predictable - i.e. it will be similar to the accuracy of validation data-set, using the training data-set.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

My pre-processing includes 2 stages:

1.- **Grayscale:** 

The raw images are in 3 channels, one per color (i.e. RGB: Red, Green, and Blue). Grayscaling it converts it to one one channel, where the information in each pixel is its intensity (degree of darkness). This makes the image have a single dimension of intensity for each pixel, which can be eaily read-in by the CNN, and optimized for. 

If, instead, I kept all 3 channels, they would have to be individually processed and optimized, thus requiring more processing. However, that could be required and may even be essential where grayscaling produces very inaccurate results, especially where the image is blurry/unclear, and where signs are of similar shapes, but different colors.

2.- **Normalization:**

Normalization is essentially making the mean (average) value of all pixels zero(0). The statistical definition also requires for it to have variance = 1. I have used the quick way, where I use: `(pixel - 128)/ 128` to bring the pixel values between -1.0 and 1.0

The key benefit of this "centering"/"normalization" is to aid in back-propagating, Having such scaled-centered values make it easier to perform gradient descent. Since we multiply weights and add biases, having these values centered around 0, keeps the amplification (by weight multiplication) in control, and thus our back-propagation model is more accurately tuned.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was based on Yan LeCun's LeNet, as following, with some key changes:
![LeNet Archictecture][image10]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscaled image  
| _Layer 1:_ Convolution 5x5   | 1x1 stride, Valid-padding, outputs 28x28x36 
| _Layer 1:_ RELU					| Rectified Linear Unit Function on above convolution
| _Layer 1:_ Max pooling	      	| 2x2 stride,  outputs 14x14x26
| _Layer 2:_ Convolution 5x5   | 1x1 stride, Valid-padding, outputs 10x10x48 
| _Layer 2:_ RELU					| Rectified Linear Unit Function on above convolution
| _Layer 2:_ Max pooling	      	| 2x2 stride,  outputs 5x5x48
| _Layer 2:_ Dropout	      	| Dropout 25% (Keep 75%)
| _Layer 2:_ Flattening	      	| Flatenning to feed into Fully-Connected Layers (5x5x48 = 1200)
| _Layer 3:_ Fully-Connected	      	| Input: 1200, Output: 240
| _Layer 3:_ RELU	      	| Rectified Linear Unit Function on above Fully-Connected output
| _Layer 4:_ Fully-Connected	      	| Input: 240, Output: 84
| _Layer 4:_ RELU	      	| Rectified Linear Unit Function on above Fully-Connected output
| _Layer 4:_ Dropout	      	| Dropout 25% (Keep 75%)
| _Layer 5:_ Fully-Connected	      	| Final layer: Input: 84, Output: 43


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:

* Epochs: 20
* Batch-size: 128
* Learning-rate: 0.001
* Optimizer: "AdamOptimizer", which implements the [Adam algorithm](https://arxiv.org/abs/1412.6980). Essentially it performs gradient-descent, and optimizes the provided loss-function, which in my case, is described next.
* Loss-Calculator: "reduce_mean", of cross entropy of logits. Essentially, we are trying to reduce the average loss in every iteration. The cross entropy represents the 'score' function, but for a classification problem. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

To achieve a validation-set accuracy of at least 0.93, I tried the following:

* Increasing the number of Epochs
* Increasing and decreasing the Batch size
* Increasing and decreasing the Learning rate
* Adding dropouts at various stages - first, after Convolutions are done, second one to strengthen final output
* Drop out value different for Validation vs Training
* Changing the Convolution layers' depths

My final model results were:

* training set accuracy of: _99%_
* validation set accuracy of: _97%_ 
* test set accuracy of: _95%_

To get there, an iterative approach was chosen.

* What was the first architecture that was tried and why was it chosen?
	* The first architecture was very simialr to the one in the lectures for LeNet. The image was grayscaled and therefore made single-channel, and it was fed into the Network. 
	* First, It was chosen because it had demonstrated that it worked with image data. 
	* Second, CNNs inherently start by picking up larger and then later smaller components of data, by convolutions, which works very well with image recognition, because CNNs encapsulate recognition of slices and chunks of images. The LeNet architecture has 2 layers of comvolution, which do just that, and therefore it was chosen.
* What were some problems with the initial architecture?
	* The initial accuracy was very low. With 10 Epochs, it could hardly reach 70% Validation Accuracy. In addition, it would hover back-and forth between 60% - 70%, thus making it unstable.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

	* The following adjustments were made:
		* **Adding dropouts at various stages:** After intial model was trained, the 60% validation-set accuracy hinted that the model was not even strong for the validation set. To strengthen it, by making it more resilient to changing inputs, I added dropouts, as follows:
			* First dropout, after Layer 1 and Layer 2 Convolutions are done: I added dropout here because at this point both the convolutions are done, and now are about to be fed into the Fully-Connected layers. This means that we have fully convoluted all the data we recieved, but we drop it before feeding it into the non-convolution layers. This was important because, at this point, the data does not relate to previous layers, because it is not a scan of the previous layers. 
				* _This improvement made the validation accuracy to go upto about 75%-80%_
			* Second dropout, just before the final layer: I added this dropout to strengthen final output. After the Fully-Connected Layer 3 and Layer 4, the data is about to be compressed into the final set of labels (43). Dropping it here would mean the model would be more resilitent to all the transformations that happened until now.
				* _This improvement made the validation accuracy to go upto about 80%-82%_
		* **Drop out value different for Validation vs Training**
			* This was more of a careless mistake. I had initially kept the dropout keep-value to 75% for both the training phase and the validation phase. Once I made the validation phase dropout keep-value to 100% for validation, the validation accuracy improved further to about 85%.

* Which parameters were tuned? How were they adjusted and why?
	* **Convolution layers' depths**
		* This single change brought my accuracies above 93%. This was a very intuitive adjustment. I realized that the LeNet Architecture that I had witnessed, was only classifying single-digit numbers. However, this project required classifying between images, which were complex symbols. Therefore I decided to increase the complexity by either: 
			* 1. Adding additional convolution layers, or
			* 2. Increasing the depth of filters
		* Since increasing the depth was more straight-forward, I did that, and it helped tremendously. The mere realization that increaseing the number of depth-filters would increase the model's interpretation of the images was profound. I increased:
			* Layer 1 depth from 6 (default) to 36 (6 times), and 
			* Layer 2 from 16 (default) to 48 (3 times).
			* _These improvements finally made the validation accuracy to go upto about 95%-97%_
	* Number of Epochs
		* This was a minor improvement, because I wanted to ensure that I was not under-tuning the model by not running it for enough Epochs. So for personal confirmation, I increased the number of Epochs to 20, to ensure that for last few epochs, the validation accuracy does not improve any further.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
	* As described above, Convolution layers scan the slices (small parts) and chunks (larger parts) of data, so they work great for image classification.
	* Dropout is just a method of regularization, and we are trying to strengthen the model by ensuring that we are not over-fitting. With classification problems, overfitting leads to high-probabilities for multiple classification, which needs to be reduced.
	* Therefore, choosing 2 convolution layers, with significant depths, feeding them later into fully-connected layers, and introducing dropout was the approach that worked well with this problem.

If a well known architecture was chosen:

* What architecture was chosen?
	* LeNet (as described above)
* Why did you believe it would be relevant to the traffic sign application?
	* As described above, it was proven to work with image data.
	* More importantly, as also described above, it fits the needs of an image-classification problem very well.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
	* In this model, we have: Training accuracy (99%) > Validation Accuracy (97%) > Test Accuracy (95%)
	* This means that the wild-set of test-data is least accuracte, but faily accurate. More importantly, since we validated it with the validation set, the validation set has higher accuracy. Finally, the training itself was done with the training set, which has the highest accuracy. Therefore, the accuracies obtained provide strong evidence that the model is working well. 
 
#####Additional section: Further improvements:
If I had more time, and some more opportunity to research, I would like to:
1. Add additional convolution layers
2. Work individually with all 3 channels (as opposed to single Gray-channel)
3. Superimpose/Concatenate Convolution-Layer-1 and Convolution-Layer-2 outputs & then passing it to the Fully-Connected layer as described in [this paper in Fig. 2](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![30 kmph Speed Limit - 1][image5] ![Children Crossing - 28][image6] ![End all limits - 32][image7] 
![No Entry - 17][image8] ![Yield - 13][image9]

Only the second (label 28) and the third (label 32) images would be hard to classify, beucase the data available for those labels is less - as shown in the histograms above.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 kmph Stop Sign      		| 30 kmph Stop Sign									| 
| Children Crossing     			| Children Crossing 										|
| End all limits					| End all limits											|
| No Entry	      		| No Entry					 				|
| Yield			| Yield


The model was able to correctly guess **all 5 of the 5 traffic signs**, which gives an accuracy of **100%.** This compares favorably to the accuracy on the test set of 95%. That means the model is fairly accurate, and given crisp-clear test-data with very little ambiguity, the model should be able to perform very well.

Another reason it was able to perform so well, is perhaps because I obtained fairly frontal-aimed images, which capture the traffic-sign in its entirety, and very clearly. None of these signs are bent, or photographed from an angle, and they are all right in the middle of the image. 
So, the model obtained very crisp data.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 21st cell of the Ipython notebook, which has a heading: **"Predict the Sign Type for Each Image"**

For the first image, 
![30 kmph Speed Limit - 1][image5]
the model is very certain that this is a 30 kmph speed-limit sign (probability of 1.0), and the image does contain a 30 kmph speed-limit sign very clearly. The top five soft max probabilities were:

| Probability         	|     Prediction	       
|:---------------------:|:---------------------------------------------:| 
| 1.00         			 | Label-1 (30 kmph speed-limit)
| 4.51 e-26     				| Label-2 (50 kmph speed-limit)
| 1.42 e-30					| Label-0 (20 kmph speed-limit)
| 8.14 e-31	      			| Label-4 (70 kmph speed-limit)
| 1.64 e-31				    | Label-5 (80 kmph speed-limit)

It is interesting to note here that the 3 highest probilities are all speed-limit signs, of 2-digit speed-limits, ending in zero (0), i.e. they are visually similar.

For the second image, ![Children Crossing - 28][image6] 
the model is again very certain that this is a children-crossing sign (probability of 0.99), even though the image is very-slightly blurry. The top five soft max probabilities were:

| Probability         	|     Prediction	       
|:---------------------:|:---------------------------------------------:| 
| 0.99         			 | Label-28 (Children crossing)
| 2.56 e-6     				| Label-29 (Bicycles crossing)
| 1.03 e-7					| Label-11 (Priority road)
| 4.52 e-8	      			| Label-18 (General caution)
| 2.66 e-9				    | Label-27 (Pedestrians)

Again, the first 2 signs (highest probabilties) are both crossings, therefore visually similar. But the model still predicted correctly.

For the third image, ![End all limits - 32][image7] 
the model is again very certain that this is a end-all-limits sign (probability of 0.99), even though the image has a white cloud above it, making it slightly ambiguous. The top five soft max probabilities were:

| Probability         	|     Prediction	       
|:---------------------:|:---------------------------------------------:| 
| 0.99         			 | Label-32 (End of all speed and passing limits)
| 4.08 e-7     				| Label-6 (End of speed limit (80km/h))
| 3.40 e-7					| Label-41 (End of no passing)
| 3.87 e-9	      			| Label-38 (Keep right)
| 4.34 e-10				    | Label-1 (Speed limit (30km/h))


For the fourth image, ![No Entry - 17][image8] 
the model is again very certain that this is a No-Entry sign (probability of 0.99). The top five soft max probabilities were:

| Probability         	|     Prediction	       
|:---------------------:|:---------------------------------------------:| 
| 0.99         			 | Label-17 (No entry)
| 2.68 e-7     				| Label-9 (No passing)
| 3.61 e-11					| Label-41 (End of no passing)
| 3.83 e-12	      			| Label-34 (Turn left ahead)
| 3.12 e-12				    | Label-35 (Ahead only)


For the fifth image, ![Yield - 13][image9]
the model is again very certain that this is a Yield sign (probability of 0.99). The top five soft max probabilities were:

| Probability         	|     Prediction	       
|:---------------------:|:---------------------------------------------:| 
| 1.00         			 | Label-13 (Yield)
| 8.59 e-16     				| Label-5 (Speed limit (80km/h))
| 1.40 e-16					| Label-1 (Speed limit (30km/h))
| 1.27 e-16	      			| Label-36 (Go straight or right)
| 2.69 e-17				    | Label-7 (Speed limit (100km/h))


In summary, most of the top probabilities were 99% or 100%, which means that the model is fairly confident in classification. The runner-ups were at least a few magnitudes of 10s behind the highest predicted label.

