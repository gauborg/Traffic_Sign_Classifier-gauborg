# Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

---

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires the following libraries:

* [TensorFlow](https://www.tensorflow.org/install)
* [Numpy](https://numpy.org/doc/1.18/user/install.html)
* [OpenCV](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
* [Pickle](https://pypi.org/)

You can also download the complete package from the following link -

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
---

## Rubric Points

### 1. Submit all relevant files.

Find the following files in the project repository.

[Traffic_Light_Classifier.ipynb](https://github.com/gauborg/Traffic_Sign_Classifier-gauborg/blob/master/Traffic_Sign_Classifier.ipynb)

[HTML Output]()


### 2. Dataset Exploration

**Summary**

I have written code that prints basic summary of our dataset. The code in cell 2 prints the various numbers of training, validation and test examples.

```output
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes/labels = 43
```

The file *signnames.csv* contains a list of signs with their labels. In the next cell (cell 3), I included code to convert our csv file *signnames.csv* to a dictionary in which keys point to signnames. This step is not mandatory as you can always refer to the label result from the .csv file. However, this enhances the identification process.

In the next cell, I have included code that randomly selects 5 images from the training dataset along with their labels. Here is a snapshot -



**Exploratory Visualization**

I have included a few images which show us the classification of training, validation and test datasets. The bar charts show how many sample images are present for each label in all these datasets.

