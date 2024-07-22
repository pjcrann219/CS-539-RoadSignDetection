# CS-539-RoadSignDetection
CS-539 Final Project

By: Codey Battista, Avi Bissoondial, Nick Chantre, Paul Crann

# I. Problem Overview
To ensure safety and effectiveness, autonomous vehicles are ordered to obey the rules of the road. Doing so requires the vehicles to observe and adhere to stop signs, traffic lights, speed limits, or any other signage in unmapped environments. Our goal is to develop a convolutional neural network (CNN) to classify traffic signage into several classes.

# II. Dataset

In this project we utilized the full Road Sign Detection Dataset from Kaggle https://www.kaggle.com/datasets/andrewmvd/road-sign-detection consisting of 4 classes of images of different road signs. Additionally we incorporated images from the GRSTB dataset which is a well known dataset for various German road sign images. This diverse dataset consists of 43 different classes, however we selected 4 of which were consistent with the classes of our other dataset to compare the performance. The classes of sign we used were stop signs, general caution signs, crosswalk signs, traffic lights, and speedlimit signs. 


# III. Methodology

## Model
![Model Diagram](assets/Diagram2.png)

The overall arcitecture of our model is a standard convolutional neural network with three convolutional layers, three max pooling layers, and two fully connected layers. In each of the convolutional layers, the ReLU activation function was used, normalization was applied, and a dropout with a probability of 0.3 was added. A droupout was also performed before each of the fully connected layers. The dropout layers were included to help reduce overfitting by forcing the model to weigh the more robust features heavier. As seen in the graph above, each convolutional layer will provide double the kernels or filters. The first convolutional layer has 32 filters, the second has 64 filters, and the third has 128 filters. The first fully connected layer has 2048 neurons, the second with 256, and the third has 5 for each of the included classes. 

## Hyper Parameters

# IV. Results
