#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:40:16 2019

@author: aramos
"""

from Image_treatment import *
#%%
#donwload dataframe
mnist_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv", sep=",",header=None)

#%%
#use only 10000 raws
mnist_dataframe = mnist_dataframe.head(10000)

#shuffle data
mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))

#%%
#First look to the datataframe
dataframe_view(mnist_dataframe)

print('The first column represent the target and the other ones are the pixels in format of 0-255 gray scale \n------------')
#%%

    
feature_dataframe, target_serie = preprocess_data_raws_255( mnist_dataframe)

#View of the feature_dataframe
dataframe_view(feature_dataframe)
#%%

plot_raw_of_dataframe_as_image(feature_dataframe, target_serie, 2)

#%%

construct_feature_columns_for_pixel_image_given_raws(feature_dataframe)

#%%
dataframe = feature_dataframe
estimator = 'DNNClassifier'
learning_rate = 0.05
clip_gradient = 5
optimizer = 'Adagrad'
n_classes = 10
hidden_units = [10, 10]
regularization = 'L1'
regularization_strength = 0.005

model_creator_for_Image(dataframe, estimator, learning_rate, 
                            clip_gradient, optimizer,  n_classes , hidden_units = hidden_units,
                            regularization = regularization,  regularization_strength = regularization_strength)


#%%

steps = 10
periods = 10
percent_training_data = 80
batch_size = 100
buffer_size = 1000
estimator = 'LinearClassifier'
learning_rate = 0.05
clip_gradient = 5
optimizer = 'Adagrad'
n_classes = 10

training_model_Image_classification(steps, periods, feature_dataframe, target_serie, percent_training_data,
                                        batch_size, buffer_size,
                                        estimator, learning_rate, clip_gradient, optimizer,  
                                        n_classes , hidden_units = None,
                                        regularization = 'L2',  regularization_strength = None)
































