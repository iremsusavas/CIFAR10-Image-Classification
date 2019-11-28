#Import Libraries / Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from keras.datasets import cifar10
(X_train,y_train) , (X_test,y_test) = cifar10.load_data()

X_train.shape
X_test.shape

#Visualize Dataset
img =random.randrange(1,len(X_train),1)
plt.imshow(X_train[img])
print(y_train[img])


width = 15
height = 15
fig,axes = plt.subplots(height,width,figsize =(25,25))   
axes = axes.ravel()
plt.subplots_adjust(hspace = 0.4)

for i in np.arange(0 , height*width):
    index = np.random.randint(0,len(X_train))
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off')
    

#Data Preparation
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
    
num_of_categories = 10

import keras
y_train = keras.utils.to_categorical(y_train , num_of_categories)
y_test = keras.utils.to_categorical(y_test,num_of_categories)

X_train = X_train/255
Input_shape = X_train.shape[1:]

#Train The Model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

cnn_model = Sequential()
cnn_model.add(Conv2D(filters=32,kernel_size=(2,2),activation='relu',input_shape=Input_shape))
cnn_model.add(Conv2D(filters=32,kernel_size=(2,2),activation='relu'))
cnn_model.add (MaxPooling2D(pool_size=2))
cnn_model.add(Dropout(0.3))

cnn_model.add(Conv2D(filters=64,kernel_size=(2,2),activation='relu',input_shape=Input_shape))
cnn_model.add(Conv2D(filters=64,kernel_size=(2,2),activation='relu'))
cnn_model.add (MaxPooling2D(pool_size=2))
cnn_model.add(Dropout(0.2))

cnn_model.add(Flatten())

cnn_model.add(Dense(units = 512,activation ='relu'))
cnn_model.add(Dense(units = 512,activation = 'relu'))
cnn_model.add(Dense(units = 10,activation ='softmax'))
cnn_model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.rmsprop(lr=0.001),metrics=['accuracy'])

fit_model = cnn_model.fit(X_train,y_train,batch_size=32,epochs=2,shuffle=True)
 
#Evaluate The Model
evaluation = cnn_model.evaluate(X_test,y_test)
print('Test Accuracy: {}'.format(evaluation[1]))

predicted_classes= cnn_model.predict_classes(X_test)
predicted_classes
y_test
#y_test =y_test.argmax(1)
#y_test

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test,predicted_classes)
cm

plt.figure(figsize=(10,10))
sns.heatmap(cm,annot = True)

#Improving The Model
#Data Augmentation
''' import keras
from keras.datasets import cifa10
(X_train,y_train),(X_test,y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

from keras.preprocessing.image import ImageDataGenertor
datagen_train = ImageDataGenerator(rotation_range=90)
datagen_train = ImageDataGenerator(vertical_flip=True)
datagen_train = ImageDataGeneraor(height_shift_range =0.5)
datagen_train = ImageDataGenerator(brightness_range=(1,5))

datagen_train.fit(X_train)
'''
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
        )
datagen.fit(X_train)
from math import ceil
n = len(X_train)
batch_size=32
steps_per_epoch = ceil(n/batch_size)
cnn_model.fit_generator(datagen.flow(X_train,y_train,batch_size=32),epochs=2,steps_per_epoch=steps_per_epoch)

reevaluate = cnn_model.evaluate(X_test,y_test)
print('Test Accuracy{}'.format(reevaluate[1]))


