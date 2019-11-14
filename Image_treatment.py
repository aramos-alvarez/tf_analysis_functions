#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:28:18 2019

@author: aramos
"""
# =============================================================================
# This scripts of functions is for the treatment of images given in a dataframe:

    #Given a dataframe with the first column as target and every raw represents a 
    #image of the pixel in 255 format gray scale
    
    #This is for Image Classification
    
# =============================================================================




import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics, preprocessing
from IPython import display
import pandas as pd
from tensorflow.python.data import Dataset
import seaborn as sns
import time
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os

#os.chdir('/home/aramos/Desktop/MachineLearning_Learning/Tensor_flow_1/user_functions')


def dataframe_view(dataframe):
    print('-----------Display Head of Raw----------')
    display.display(dataframe.head())
    print('\n-----------Description---------')
    display.display(dataframe.describe())



def preprocess_data_raws_255(dataframe):
    #  The first column represent the target and 
    # the other ones are the pixels in format of 0-255 gray scale 
    # I should separte in dataframe for training and for target
    
    
    target_serie = dataframe.loc[:, 0]
    feature_dataframe = dataframe.loc[:, 1:] / 255 #normalize the grayscale between 0 and 1
    return feature_dataframe, target_serie
    







def plot_raw_of_dataframe_as_image(feature_dataframe, target_serie, how_many):
    #Given a dataframe with pixels of a image in every raw this function can plot as images
    #See the example for HandWritter given in this example
    
    plt.close('Image from random raw feature_df')
    figure_feature_df = plt.figure('Image from random raw feature_df')
    ax_feature_df = figure_feature_df.add_subplot(1,1,1)
    
    for period in range(how_many+1):
        
        ax_feature_df.clear()
        random_raw = np.random.choice(feature_dataframe.index)
        picture = np.array(feature_dataframe.loc[random_raw, :])
        picture = picture.reshape(int(np.sqrt(picture.size)), int(np.sqrt(picture.size)))
        target_variable = target_serie.loc[random_raw]
        ax_feature_df.set_title(str(target_variable))
        ax_feature_df.imshow(picture, interpolation = 'gaussian')
        figure_feature_df.canvas.draw()
        plt.pause(0.05)
        time.sleep(1)
        
    return ax_feature_df



def my_input_fn_image_in_raws(feature_dataframe, target_serie, batch_size, num_epochs, shuffle, buffer_size):
    
    
    feature = {'pixels': feature_dataframe.values}
    ds = Dataset.from_tensor_slices((feature, target_serie))
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    if shuffle:
        ds = ds.shuffle(buffer_size)
    
    
   
    return ds.make_one_shot_iterator().get_next()


    
def construct_feature_columns_for_pixel_image_given_raws(dataframe):
    #Construct the feature columns from images given as raws in a dataframe
    
    return set([tf.feature_column.numeric_column('pixels', shape = dataframe.shape[1])])



def optimizer_creator(learning_rate, clip_gradient, optimizer = 'GradientDescent', regularization = 'L2', 
                     regularization_strength = None):
    
    
    #optimizer could be: Adagrad, Ftrl, GradientDescent
    
    
    #L2 Regularization
    if regularization == 'L2':
    
    
        #Adagrad
        try: 
            #optimizer
            if optimizer == 'Adagrad':
                my_optimizer = tf.train.AdagradOptimizer(learning_rate)

            elif optimizer == 'Ftrl':
                my_optimizer = tf.train.FtrlOptimizer(learning_rate)

            elif optimizer == 'GradientDescent':
                my_optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, clip_gradient)


            print('Optimizer used: '  + optimizer)


        except (ValueError, UnboundLocalError):
            print('Optimizer Bad Defined !! \n------------\n consider some of: Adagrad, Ftrl, GradientDescent')
            my_optimizer = None
    
    
    #L1 regularization
    elif regularization == 'L1':
                
        try: 
                
            optimizer = 'FtrlOptimizer'
            my_optimizer = tf.train.FtrlOptimizer(learning_rate, l1_regularization_strength =
                                                         regularization_strength)

            print('Optimizer used with L1 regularization (only valid): '  + optimizer)

        except (ValueError, UnboundLocalError):
            print('Optimizer Bad Defined !! \n------------\n consider some of: Adagrad, Ftrl, GradientDescent')

    #Bad defined the regularization
    else:
        print('Bad defined the regularization, try: L1 or L2')
        my_optimizer = None
            
            
    return my_optimizer





def model_creator_for_Image(dataframe, estimator, learning_rate, 
                            clip_gradient, optimizer,  n_classes , hidden_units = None,
                            regularization = 'L2',  regularization_strength = None):
    
    #estimator should be: LinearRegressor, LinearClassifier, DNNRegressor (in the case of Neural Network)
    my_optimizer = optimizer_creator(learning_rate, clip_gradient, optimizer, regularization, 
                     regularization_strength)
    
    
    
    #chekc if all elements in list numeric and list_categorical are in the dataframe

    

    feature_columns = construct_feature_columns_for_pixel_image_given_raws(dataframe)
    
    
    print('Feature Columns:  \n -----------')
    display.display(feature_columns)
    
    if estimator == 'LinearClassifier':
        print('\n Estimator:   ' + estimator)
        model = tf.estimator.LinearClassifier(feature_columns = feature_columns, n_classes = n_classes, 
                                              optimizer = my_optimizer,
                                              config = tf.contrib.learn.RunConfig(keep_checkpoint_max = 1))
        
        
    elif estimator == 'DNNClassifier':
        print('\n Estimator:   ' + estimator + ' with hidden_units:...' + str(hidden_units))
        model = tf.estimator.DNNClassifier(feature_columns = feature_columns,  n_classes = n_classes,
                                           hidden_units = hidden_units, 
                                          optimizer = my_optimizer,
                                          config = tf.contrib.learn.RunConfig(keep_checkpoint_max = 1))
        
        
    
    else:
        print('\n BAD DEFINED ESTIMATOR, use: LinearClassifier or DNNClassifier')
    

    return model


#%%
    
def training_model_Image_classification(steps, periods, feature_dataframe, target_serie, percent_training_data,
                                        batch_size, buffer_size,
                                        estimator, learning_rate, clip_gradient, optimizer,  
                                        n_classes , hidden_units = None,
                                        regularization = 'L2',  regularization_strength = None):
    
    # This function trains a model for Image Classification problem, the images should provide in the dataframe in 
    # every raw
    
    
    steps_per_period = steps / periods
    
    
    total_raws = feature_dataframe.count()[1]
    
    #DATAFRAME FOR TRAINING
    feature_dataframe_training = feature_dataframe.head(int(percent_training_data/ 100 * total_raws))
    target_serie_training = target_serie.head(int(percent_training_data / 100 * total_raws))
    
    #DATAFRAME FOR VALIDATION
    feature_dataframe_validation = feature_dataframe.tail(int(1-(percent_training_data/ 100) * total_raws))
    target_serie_validation = target_serie.tail(int(1-(percent_training_data/ 100) * total_raws))




    #train input fn
    num_epochs = None
    shuffle = True

    train_input = lambda: my_input_fn_image_in_raws(feature_dataframe_training , target_serie_training,
                              batch_size, num_epochs, shuffle, buffer_size)
    
    #predict training input fn
    num_epochs = 1
    shuffle = False
    batch_size = 1
    
    predict_training_input = lambda: my_input_fn_image_in_raws(feature_dataframe_training , target_serie_training, 
                                         batch_size, num_epochs,shuffle, buffer_size)
    
    
    
    #predict validation input fn
    predict_validation_input = lambda: my_input_fn_image_in_raws(feature_dataframe_validation , target_serie_validation, 
                                         batch_size, num_epochs,shuffle, buffer_size)
    

    model_image_classification = model_creator_for_Image(feature_dataframe, estimator, learning_rate, 
                            clip_gradient, optimizer,  n_classes , hidden_units ,
                            regularization,  regularization_strength)
    
    
    #FIGURE CONSTRUCTOR FOR THE TRAINING DATA
    plt.close('TRAINING MODEL')
    figure_training = plt.figure('TRAINING MODEL', figsize = (8,8))
    ax_training = figure_training.add_subplot(1,1,1)

    log_loss_training_list = []
    log_loss_validation_list = []
    #TRAINING THE MODEL BY STEPS
    print('TRAINING MODEL...\n')
    
    if estimator == 'LinearClassifier' or estimator == 'DNNClassifier':
        print('Minimize Log Loss function (clasification problem) \n')
        
        #Creating a Encoder for classification problem
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(target_serie_training)
        
        
    else:
        print('ERROR defining estimator')
        
    for period in range(0, periods):
        
        
        _ = model_image_classification.train(input_fn = train_input, steps = steps_per_period)
        
        #take a break to predict values and calculate the error
        prediction_training = model_image_classification.predict(input_fn = predict_training_input)
        prediction_validation = model_image_classification.predict(input_fn = predict_validation_input)
        
        ax_training.set_title('Log Loss')
        
        prediction_training = np.array([item['class_ids'][0] for item in prediction_training])
        prediction_validation = np.array([item['class_ids'][0] for item in prediction_validation])
        
        #In the case of the output is a string I should preprocess the data before to convert to 
        #one hot
        #Encoding the prediction
        prediction_training_encoded = label_encoder.transform(prediction_training)
        prediction_validation_encoded = label_encoder.transform(prediction_validation)
        
        
        
        #convert the prediction in one hot encoding ---> to categorical
        prediction_training_one_hot = tf.keras.utils.to_categorical(prediction_training_encoded,
                                                                     n_classes)
        prediction_validation_one_hot = tf.keras.utils.to_categorical(prediction_validation_encoded,
                                                                      n_classes)
        
        
        
        log_loss_training = metrics.log_loss(target_serie_training, prediction_training_one_hot)
        log_loss_training_list.append(log_loss_training)
        

        
        log_loss_validation = metrics.log_loss(target_serie_validation, prediction_validation_one_hot)
        log_loss_validation_list.append(log_loss_validation)
        
        print('PERIOD ' + str(period) + '...log_loss_training: ' + str(np.round(log_loss_training,4)))
        
        
        #plotting
        ax_training.clear()
        ax_training.plot(log_loss_training_list, '-o', color = 'blue', label = 'Training data')
        ax_training.plot(log_loss_validation_list, '-x', color = 'red', label = 'Validation data')
        
        figure_training.canvas.draw()
        plt.pause(0.01)

    ax_training.legend()        
    
    
    print('MODEL TRAINED!!')
    return model_image_classification



















