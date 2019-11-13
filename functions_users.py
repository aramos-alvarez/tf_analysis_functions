#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:00:02 2019

@author: aramos
"""

import os

os.chdir('/home/aramos/Desktop/MachineLearning_Learning/Tensor_flow_1/user_functions')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics, preprocessing
from IPython import display
import pandas as pd
from tensorflow.python.data import Dataset
import seaborn as sns

tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def dataframe_view(dataframe):
    print('-----------Display Head of Raw----------')
    display.display(dataframe.head())
    print('\n-----------Description---------')
    display.display(dataframe.describe())



def correlation_figure(dataframe):
    print('Data correlation matrix')
    correlation = dataframe.corr()
    plt.close('Matrix Correlation')
    figure_corr = plt.figure('Matrix Correlation')
    ax_corr = figure_corr.add_subplot(1,1,1)
    _ = sns.heatmap(correlation, ax = ax_corr, annot = True)
    figure_corr.canvas.draw()
    return ax_corr


def scatter_matrix(dataframe):
    plt.close('Scatter Matrix')
    print('Scatter Matrix')
    figure_scatter = plt.figure('Scatter Matrix', figsize = (10,10))
    ax_scatter = figure_scatter.add_subplot(1,1,1)

    _ = pd.plotting.scatter_matrix(dataframe, ax = ax_scatter)
    figure_scatter.canvas.draw()
    return ax_scatter


#preprocess data

def scaling_serie(serie):
    serie_min = serie.min()
    serie_max = serie.max()
    scale = abs(serie_max - serie_min)
    serie = serie.apply(lambda x: 2*((x - serie_min) / scale - 0.5))
    
    return serie


def scaling_dataframe(dataframe):
    feature_names = list(dataframe.keys())
    
    for feature_name in feature_names:
        dataframe[feature_name] = scaling_serie(dataframe[feature_name])
    
    return dataframe
    


def preprocess_data(dataframe, target_name):
    
    dataframe_temp = dataframe.copy()
    target = dataframe_temp.pop(target_name) / 1000
    dataframe_temp['rooms_per_person'] = dataframe_temp['total_rooms'] / dataframe_temp['population']
    dataframe_temp = scaling_dataframe(dataframe_temp)
    
    return dataframe_temp, target


target_name = 'median_house_value'
feature_dataframe, target_serie = preprocess_data(california_housing_dataframe, target_name)


def preprocess_data_for_classifier(dataframe, target_name, threshold):
    
    dataframe_temp = dataframe.copy()
    target = (dataframe_temp.pop(target_name) > threshold).astype(float)
    dataframe_temp['rooms_per_person'] = dataframe_temp['total_rooms'] / dataframe_temp['population']
    dataframe_temp = scaling_dataframe(dataframe_temp)
    
    return dataframe_temp, target


# prepare the input function
def my_input_fn(feature_dataframe, target_serie, batch_size, num_epochs, shuffle, buffer_size):
    
    
    
    ds = Dataset.from_tensor_slices((dict(feature_dataframe), target_serie))
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    if shuffle:
        ds = ds.shuffle(buffer_size)
    
    
   
    return ds.make_one_shot_iterator().get_next()




def get_boundaries_quantile(serie, num_buckets):
    boundaries = np.arange(0, num_buckets) / num_buckets
    quantile_boundaries = list(serie.quantile(boundaries))
    return quantile_boundaries



def construct_feature_columns(dataframe, estimator,
                              list_numeric = None, bucketize = False, dict_variables_bucketized = None, 
                              list_categorical = None, list_crossed_features = None, hash_bucket_size = None):
    
    
    #list_crosses_features = [['median_house_value', 'population'], ['population', 'high_price']] --->NOTE DOBLE [[], []]
    #IF bucketize = True you should provide a dictionary saying if the variable is bucketize or not and the
    #number of the buckets as below:
    
    # example of dict_variables_bucketized = {'total_rooms': 10, 'population': 8, 'median_house_value': 10 }
    #list of feature tf.feature_column.numeric_column or tf.feature_column.categorical_column_with_vocabulary
    
    #hash_bucket_size should be a list with the hash_bucket_size of every crossed variables ex. [100, 200]
    feature_column = []
    
    dict_of_features_numericals = {}
    dict_of_features_bucketized_ready = {}
    dict_of_features_categoricals = {}
    
    #Feature Numeric
    if list_numeric != None:
        
        if bucketize == True:

            for feature_numeric_name in list_numeric:

                try: 
                    feature_numeric = tf.feature_column.numeric_column(feature_numeric_name)

                    feature_numeric_bucketized = tf.feature_column.bucketized_column(feature_numeric, 
                                                                                    boundaries = 
                                                     get_boundaries_quantile(dataframe[feature_numeric_name], 
                                                        dict_variables_bucketized[feature_numeric_name]))

                    print('Feature ' + feature_numeric_name + ': has been bucketized by ' + 
                          str(dict_variables_bucketized[feature_numeric_name])
                          + ' quantiles')
                    
                    #save the features bucketized into a dict
                    
                    dict_of_features_bucketized_ready[feature_numeric_name + '_bucketized'] = feature_numeric_bucketized
                                                                            
                                                                                                        

                    feature_column.append(feature_numeric_bucketized)
                
                except KeyError:
                    feature_numeric = tf.feature_column.numeric_column(feature_numeric_name)
                    feature_column.append(feature_numeric)
                    dict_of_features_numericals[feature_numeric_name] = feature_numeric
            
            #Show the dictionary with the bucketized features
            
            print('NUMERICAL FEATURES CONSIDERED: \n..........') 
            print('\n ---------Numerical features Bucketized:   ')
            print(dict_of_features_bucketized_ready.keys())
            print('\n ---------Numerical features Not Bucketized:   ')
            print(dict_of_features_numericals.keys())
        else:
            for feature_numeric_name in list_numeric:
                
                feature_numeric = tf.feature_column.numeric_column(feature_numeric_name, 
                                                                       dtype = tf.float32)
                feature_column.append(feature_numeric)
                dict_of_features_numericals[feature_numeric_name] = feature_numeric

            print('NUMERICAL FEATURES CONSIDERED: \n.............  ')
            print(dict_of_features_numericals.keys())
            
    #Feature categorical
    
    if list_categorical != None:
        
        print('\n CATEGORICAL FEATURES CONSIDERED: \n--------------')
        for feature_categorical_name in list_categorical:
            vocabulary = list(dataframe[feature_categorical_name].unique())
            print('------------')
            print('Categorical variable... ' + feature_categorical_name + ' ....has a vocabulary of:\n' )
            print(vocabulary)
            print('------------')
            feature_categorical = tf.feature_column.categorical_column_with_vocabulary_list(feature_categorical_name,
                                                                                           vocabulary)
            
            if estimator == 'DNNRegressor':
                feature_categorical_embedded = tf.feature_column.embedding_column(feature_categorical, dimension = 
                                                                        len(vocabulary))
            
                feature_column.append(feature_categorical_embedded)
            else:
                feature_column.append(feature_categorical)
            
            dict_of_features_categoricals[feature_categorical_name] = feature_categorical
        
        
        print(dict_of_features_categoricals.keys())
        print('\n --------')
    
    #Feature Crosses: ONLY FEATURES BUCKETIZED CAN BE CROSSED!!

    if list_numeric == None and list_categorical == None:
        print('List of features not defined')
    if list_crossed_features != None and len(list_crossed_features) > 0:
        print('CROSSES VARIABLES: \n----------')
        hash_bucket_size = iter(hash_bucket_size) #convert the list hash_bucket in an iterator
        for features_crossed in list_crossed_features:
            
            #See if some of the features given in features_crossed is into dict_of_features_bucketized_ready
            
            features_crossed_temporal = []
            features_crossed_temporal_names = []
            for feature in features_crossed:
                
                
                #the feature can be in features bucketized
                
                if feature + '_bucketized' in dict_of_features_bucketized_ready:
                    
                    features_crossed_temporal.append(
                        dict_of_features_bucketized_ready[feature + '_bucketized'])
                    
                    features_crossed_temporal_names.append(feature + '_bucketized')
                
                #or the feature can be in categorical features
                elif feature in dict_of_features_categoricals:
                    features_crossed_temporal.append(dict_of_features_categoricals[feature])
                    features_crossed_temporal_names.append(feature)
                else:
                    print('ERROR CROSSING:.the variable...'+ feature + '..is not bucketized before \n or not is categorical!')
                    break
            hash_bucket = next(hash_bucket_size)

            
            if len(features_crossed_temporal) > 1:
                
                print('Crossing the variables (hash bucket --' + str(hash_bucket) + ')' )
            
                print(features_crossed_temporal_names)
                print('----------------')
                display.display(features_crossed_temporal)
                feature_cro = tf.feature_column.crossed_column(features_crossed_temporal, 
                                                     hash_bucket_size = hash_bucket)
                
                if estimator == 'DNNRegressor':
                    feature_cro = tf.feature_column.embedding_column(feature_cro, dimension = hash_bucket)
                feature_column.append(feature_cro)
        
    else:
        print('NOT CROSSED FEATURES')
        
    return set(feature_column)



def optimizer_creator(learning_rate, clip_gradient, optimizer = 'GradientDescent', regularization = 'L2', 
                     regularization_strength = None):
    
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
            print('Optimizer bad defined')
    
    
    #L1 regularization
    elif regularization == 'L1':
                
        try: 
                
            optimizer = 'FtrlOptimizer'
            my_optimizer = tf.train.FtrlOptimizer(learning_rate, l1_regularization_strength =
                                                         regularization_strength)

            print('Optimizer used with L1 regularization (only valid): '  + optimizer)

        except (ValueError, UnboundLocalError):
            print('Optimizer bad defined')

    #Bad defined the regularization
    else:
        print('But defined the regularization, try: L1 or L2')
        my_optimizer = None
            
            
    return my_optimizer







#%%

if __name__ == '__main__':
    
    
    california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
    california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
    
#%%
    #view raws of dataframe
    dataframe_view(california_housing_dataframe)
    
    
#%%
    #Scatter and correlation Matrix
    
    
    #Correlation
    correlation_figure(california_housing_dataframe)
    
    #Scatter Matrix
    
    scatter_matrix(california_housing_dataframe)
    
    
#%%
    
    #Preparing data preprocess for Classification problem
    dataframe = california_housing_dataframe.copy()
    target_name = 'median_house_value'
    threshold = 210000
    feature_dataframe_classifier, target_serie_classifier = preprocess_data_for_classifier(dataframe, 
                                                                                       target_name, threshold)

    
    
    
    
#%%
    #dataframe with categorical to check construct columns
    dataframe_with_categorical = california_housing_dataframe.copy()
    threshold = 200000
    dataframe_with_categorical['high_price'] = (california_housing_dataframe['median_house_value'] > threshold).astype(str)
    display.display(dataframe_with_categorical.head())
    
    #check construct_feature_columns
    list_numeric = ['total_rooms', 'population', 'median_house_value', 'median_income']
    dict_variables_bucketized = {'total_rooms': 10, 'population': 8, 'median_house_value': 10 }
    
    estimator = 'LinearClassifier'
    feature_column = construct_feature_columns(dataframe_with_categorical, estimator,list_numeric = list_numeric, bucketize = True, dict_variables_bucketized = dict_variables_bucketized, 
                                  list_categorical = ['high_price'], list_crossed_features = [['median_house_value', 'population'], ['population', 'high_price']], hash_bucket_size = [100, 200])
        
    display.display(dataframe_with_categorical.head())
    display.display(feature_column)