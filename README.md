# Facial Emotion Recognition Using Deep Convolutional Neural Network

# Introduction 
The rapid growth of artificial intelligence has contributed a lot to the technology world. As the traditional algorithms failed to meet the human needs in real time, Machine learning and deep learning algorithms have gained great success in different applications such as classification systems, recommendation systems, pattern recognition etc. Emotion plays a vital role in determining the thoughts, behaviour and feeling of a human. An emotion recognition system can be built by utilizing the benefits of deep learning and different applications such as feedback analysis, face unlocking etc. can be implemented with 
good accuracy. The main focus of this work is to create a Deep Convolutional Neural Network (DCNN) model that classifies 5 different human facial emotions. The model is trained, tested and validated using the manually collected image dataset

# Proposed Model

The architecture for the proposed facial emotion recognition model is depicted in Figure 1. The model uses two convolution layers with dropouts after each convolution layer. The input image is resized to 32 x 32 and is given to the first convolution layer. The output from the convolution layer, called feature map, is passed through an activation function. The activation function used here is ReLU (Rectified Linear Unit) that makes the negative values zero while the positive values remain the same. This feature map is given to the pooling layer of pool size 2 x 2 to reduce the size without losing any information. Dropout layer is used so as to reduce the overfitting. This process again continues for the next convolution layer as well. Finally, a 2-dimensional array is created with some feature values. Flatten layer is used to convert these 2-dimensional arrays to a single dimensional vector so as to give it as the input of the neural network, represented by the dense layers. Here a two-layer neural network is used, one is input and the other is output. The output layer has 5 units, since 5 classes need to be classified. The activation function used in the output layer is softmax, which produces the probabilistic output for each class. Figure 2 depicts a snapshot of the model summary of the proposed system which is built using the Keras DL Library. 

![Screenshot_2023-06-25-22-16-58-51_e2d5b3f32b79de1d45acd1fad96fbb0f-01](https://github.com/PranavEswr/Facial-Emotion-Recognition-Using-CNN/assets/91025454/bff51aec-9023-4cb9-bc8a-85af0788255b)


![E2](https://github.com/PranavEswr/Facial-Emotion-Recognition-Using-CNN/assets/91025454/78b05d58-dda0-452b-b706-c5a7b37604ad)


# Results

CNN is trained with the emotion image dataset, utilizing Adam as the optimizer and the categorical cross-entropy as the loss function. The model parameters are shown in table below. 




Figure below depicts the normalized confusion matrix for the test samples using the proposed DCNN model. The specificity (recall) i.e. the coverage of positive samples shows that most of them are predicted as positive itself except class 1 (happy). 
Class 0 (angry) and Class 3 (neutral) are having good prediction results. 

# Conclusion
The model classifies 5 different facial emotions from the image dataset. The model has 
comparable training accuracy and validation accuracy which convey that the model is having a best fit and is generalized to the data. The model uses an Adam optimizer to reduce the loss function and it is tested to have an accuracy of 78.04%. The work can be extended to find out the changes in emotion using a video sequence which in turn can be used for different real time applications such as feedback analysis, etc. This system can also be integrated with other electronic devices for their effective control.
