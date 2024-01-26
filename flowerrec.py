# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:20:21 2024

@author: Lingaraja Thippeswamy
"""

import numpy as np 
import matplotlib.pyplot as plt 
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Flatten, Dense 
from tensorflow.keras.optimizers import Adam 

base_dir = "D:/Lingaraj/Python/flowers/data"

train_data_dir = 'D:/Lingaraj/Python/flowers/data/train'
validation_data_dir = 'D:/Lingaraj/Python/flowers/data/validation'
test_data_dir = 'D:/Lingaraj/Python/flowers/data/test'
  
img_size = 224
batch = 64

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,  
                                    zoom_range=0.2, horizontal_flip=True, 
                                    validation_split=0.2) 
  
test_datagen = ImageDataGenerator(rescale=1. / 255, 
                                  validation_split=0.2) 
  
# Create datasets 
train_datagen = train_datagen.flow_from_directory(directory=train_data_dir, 
                                                  target_size=( 
                                                      img_size, img_size)) 
test_datagen = test_datagen.flow_from_directory(directory=test_data_dir, 
                                                target_size=( 
                                                    img_size, img_size)) 


model = Sequential() 
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', 
                  activation='relu', input_shape=(224, 224, 3))) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
  
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                  padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
  
  
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                  padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
  
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                  padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(512)) 
model.add(Activation('relu')) 
model.add(Dense(5, activation="softmax")) 

model.summary()

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
epochs=5
model.fit(train_datagen,epochs=epochs,validation_data=test_datagen)

from tensorflow.keras.models import load_model 
model.save('Model.h5') 
   
# load model 
savedModel=load_model('Model.h5')


train_datagen.class_indices

from keras.preprocessing import image

list_ = ['Daisy','Danelion','Rose','sunflower', 'tulip'] 
  
#Input image 
test_image = image.load_img('D:/Lingaraj/Python/flowers/data/test/rose.jpg',target_size=(224,224)) 
  
#For show image 
plt.imshow(test_image) 
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image,axis=0) 
  
# Result array 
result = savedModel.predict(test_image) 
print(result) 
 
i=0
for i in range(len(result[0])): 
  if(result[0][i]==1): 
    print(list_[i]) 
    break



