#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer
import math


# In[2]:


import tensorflow as tf
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Activation, LSTM, Bidirectional
from tensorflow.keras.layers import Dense , Flatten , Input


# In[94]:


class Simple_model(tf.keras.Model):
    
    def __init__( self, number_of_classes = 10 ):
        super(Simple_model , self ).__init__()
        
        self.number_of_classes = number_of_classes
        
        self.conv1 = Convolution2D(filters=64 , kernel_size= 3 , strides=1 , padding='same',activation='relu')
        self.pool1 = MaxPooling2D(pool_size=( 2 , 2) , strides=2 , padding='valid')
        
        
        self.conv2 = Convolution2D(filters=128 , kernel_size=3 , strides=1 , padding='same', activation='relu')
        self.pool2 = MaxPooling2D(pool_size=(2,2), strides=2 , padding='valid')
        
        
        self.conv3 = Convolution2D(filters=256 , kernel_size=3 , strides=1 , padding='same',activation='relu', use_bias=False)
        
        
        self.conv4 = Convolution2D(filters=256 , kernel_size=3 , strides=1 , padding='same', activation='relu' , use_bias=False)
        self.pool4 = MaxPooling2D(pool_size=(2,2),strides=2 , padding='valid')
        
        
        self.conv5 = Convolution2D(filters=512 , kernel_size=3 , strides=1 , padding='same', use_bias=False)
        self.norm5 = BatchNormalization(trainable=True , scale=True)
        self.relu5 = Activation(activation='relu')
        
        
        
        # Dense Layer
        self.conv6 = Convolution2D(filters=512 , kernel_size=3 , strides=1 , padding='same', use_bias=False)
        self.norm6 = BatchNormalization(trainable=True , scale=True)
        self.relu6 = Activation(activation='relu')
        self.pool6 = MaxPooling2D(pool_size=(2,2) , strides = 2 , padding = 'valid')
        
        
        self.flatten_layer = Flatten()
        self.dense1 = Dense(64 )
        self.activation1 = Activation(activation='relu')
        self.out_put_layer = Dense(self.number_of_classes , activation='softmax')
        
        
    def call(self, inputs ):
        
        inputs = tf.cast(inputs , dtype=tf.float32)
        
        conv1 = self.conv1(inputs)
        pool1 = self.pool1(conv1)
        
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        
        
        conv4 = self.conv4(conv3)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5(conv4)
        norm5 = self.norm5(conv5)
        relu5 = self.relu5(norm5)
        
        conv6 = self.conv6(relu5)
        norm6 = self.norm6(conv6)
        relu6 = self.relu6(norm6)
        pool6 = self.pool6(relu6)
        
        flatten = self.flatten_layer(pool6)
        dense1 = self.dense1(flatten)
        relu_d = self.activation1(dense1)
        output_l = self.out_put_layer(relu_d)        
        return output_l
    
    
    def loss_function(self , real_y , pred_y ):
        return tf.keras.losses.categorical_crossentropy(real_y, pred_y)
    
    
    
    def preprocessing(self, X_data ,  Y_data):
        #Normalizing images data
        X_data = np.array(X_data).astype("float32")
        X_data = X_data/255.0
        
        # Maping label data 
        Y_data_onehotcodeing = tf.keras.utils.to_categorical(Y_data, self.number_of_classes)
            
        return ( X_data , Y_data_onehotcodeing )
    
    
    
    
    def per_step(self,  model ,optimizer ,  real_x_values , real_y_values ):
        
        with tf.GradientTape() as tape:
            pred_y_value = model(real_x_values)
            model_loss = self.loss_function(real_y_values , pred_y_value )
            
        model_gradients = tape.gradient( model_loss , model.trainable_variables )
        optimizer.apply_gradients(zip(model_gradients , model.trainable_variables))
        
    
    
    
    
    
    
    
    def train_model(self ,model , train_x_data ,train_y_data, test_x_data , test_y_data, 
                    learning_rate = 0.0001 , batch_size = 8 , number_of_epochs = 3 ):
        
        # preprocessing train and test data
        x_train , y_train = self.preprocessing( train_x_data , train_y_data )
        x_test  , y_test  = self.preprocessing( test_x_data ,  test_y_data )
        
        #Select the optimizer
        optimizers = tf.keras.optimizers.Adam(lr = learning_rate , beta_1=0.9 , 
                                      beta_2=0.99, epsilon=1e-08 ,decay = 0.0)
        
        
        batch_per_epochs = math.floor(len(x_train) / batch_size )
        
        for epoch in range(number_of_epochs ):
            print("=" , end = '')
            for i in range(batch_per_epochs):
                n = i*batch_size
                X_train_data = x_train[n : n + batch_size ]
                Y_train_data = y_train[n : n + batch_size ]
                
                self.per_step(model , optimizers , X_train_data , Y_train_data )
                
        
        # Calculate accuracy
       
                
        
    
        
        
        
        
        
        
        
        
        


# In[95]:


model = Simple_model(number_of_classes = 3 )


# In[96]:


import os , cv2


# In[97]:


images_path = "D:/temp/rlvd/training_images/"
light_colors  = ["red" , "green" , "yellow"]


# In[98]:


data = []
color_counts = np.zeros(3)


# In[99]:


for color in light_colors:
    for img_file in os.listdir(os.path.join(images_path , color )):
        img_file_path  = os.path.join(images_path , color , img_file )
        img = cv2.imread(img_file_path)
        try:
            img = cv2.resize(img , ( 32 , 32) )
            label_idx = light_colors.index(color)
            data.append((img , label_idx ))
            print(img_file_path , end = ' ' , flush=True)
           
        except :
            print("Error" , end = '\n' , flush= True)


# In[100]:


len(data)


# In[101]:


import random
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


# In[102]:


random.shuffle(data)


# In[103]:


X , Y = [] , []


# In[104]:


for sample in data:
    X.append(sample[0])
    Y.append(sample[1])


# In[105]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.086,
        random_state = 832289)


# In[106]:


model.train_model( model , X_train , y_train , X_test , y_test )


# In[ ]:





# In[ ]:




