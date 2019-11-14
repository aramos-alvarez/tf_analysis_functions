#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:35:51 2019

@author: aramos
"""

from functions_for_analysis_dataframes import *

#%%
#Donwload
#dataframe for training and test
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

#Shuffling data
dftrain = dftrain.reindex(np.random.permutation(dftrain.index))
dfeval = dfeval.reindex(np.random.permutation(dfeval.index))

#%%

#First look to the dataframe
dataframe_view(dftrain)

#%%

#Plotting the categorical variables
column_categorical_name = 'sex'
plot_categorical_counts(dftrain, column_categorical_name)


#%%
#correlation, ONLY numerical features
correlation_figure(dftrain)

#%%
scatter_matrix(dftrain)

#%%
#Preprocess data for Titanic

def preprocess_dataframe_Titanic(dataframe, target_name):
    
    dataframe_temp = dataframe.copy()
    target = dataframe_temp.pop(target_name)
    
    return dataframe_temp, target
    
feature_dataframe_Titanic, target_serie_Titanic = preprocess_dataframe_Titanic(dftrain, 'survived')
feature_dataframe_eval_Titanic, target_serie_eval_Titanic = preprocess_dataframe_Titanic(dfeval, 'survived')
dataframe_view(feature_dataframe_Titanic)



#%%





steps = 10
periods = 10
feature_dataframe = feature_dataframe_Titanic.copy()
target_serie = target_serie_Titanic.copy()
percent_training_data = 80
list_numeric = ['age', 'fare']
list_categorical = ['sex','n_siblings_spouses','parch', 'class', 'deck', 'embark_town', 'alone']  
estimator = 'DNNClassifier'
learning_rate = 0.005
batch_size = 100
buffer_size = 1000




bucketize = True 
dict_variables_bucketized = {'age': 7, 'fare': 10}                
list_crossed_features = [['sex', 'parch']]
hash_bucket_size = [100] 
optimizer = 'Adagrad'
clip_gradient = 5 
n_classes = 2
hidden_units = [10,10]
regularization = 'L2'  
regularization_strength = None



training_model(steps, periods, feature_dataframe, target_serie, percent_training_data, 
                   list_numeric, list_categorical, estimator, learning_rate, batch_size, buffer_size, 
                   bucketize, dict_variables_bucketized, 
                  list_crossed_features, hash_bucket_size, 
                  optimizer,
                   clip_gradient, 
                   n_classes,
                   hidden_units,
                  regularization,  regularization_strength)












