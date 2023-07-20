############# Import Libraries ############
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten,Dense,Dropout,Activation
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders

############# Split Dataset into Train, Test & Valid #############
splitfolders.ratio('5_Emotions/', output="splitted_data", seed=1337, ratio=(.8, .1, .1))

############# Setup the data generators #############
train = ImageDataGenerator(rescale = 1/255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

valid = ImageDataGenerator(rescale = 1/255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

batch_size  = 32
picture_size = 32

train_dataset = train.flow_from_directory("splitted_data/train",
                                              target_size = (picture_size,picture_size),
                                              color_mode = "rgb",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=True)


valid_dataset = valid.flow_from_directory("splitted_data/val",
                                              target_size = (picture_size,picture_size),
                                              color_mode = "rgb",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=True)

############# To fetch contents from directory #############
import os
path="splitted_data/test"
os.listdir(path)

############# Setup our Convolutional Neural Network (CNN) #############
from keras.optimizers import Adam

no_of_classes = 5

model = Sequential()

#1st CNN layer
model.add(Conv2D(32,(3,3), activation='relu', padding = 'same',input_shape = (30,30,1)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

#2nd CNN layer
model.add(Conv2D(32,(3,3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout (0.2))

#Flatten Layer
model.add(Flatten())

#Fully connected 1st layer
model.add(Dense(256,activation='relu'))

# Fully connected layer 2nd layer
model.add(Dense(5,activation='softmax'))

# opt = Adam(lr = 0.01)
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

filepath = 'adam_weights.hdf5'
acc_check = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
stop_check = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='max')
callbacks_list = [acc_check,stop_check]

############# Train the model #############
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history_adam = model.fit(train_dataset, 
                     steps_per_epoch=2100,
                     epochs=11,
                     validation_data = valid_dataset,
                     validation_steps=210,
                     callbacks=callbacks_list)

model.save_weights("./model.h5")

############# Analyze the results #############

# plot the evolution of Loss and Acuracy on the train and validation sets

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

plt.subplot(1, 2, 1)
plt.plot(history_adam.history['accuracy'])
plt.plot(history_adam.history['val_accuracy'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')


plt.subplot(1, 2, 2)
plt.plot(history_adam.history['loss'])
plt.plot(history_adam.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
