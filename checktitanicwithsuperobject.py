#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:35:17 2019

@author: aramos
"""

from super_object import *
plt.close('all')
#%%
#Donwload
#dataframe for training and test
print('Donwload dataframe for Titanic model')
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

#Shuffling data
dftrain = dftrain.reindex(np.random.permutation(dftrain.index))
dfeval = dfeval.reindex(np.random.permutation(dfeval.index))

#%%

# =============================================================================
# possible function to determine numericals and categorical variables
# =============================================================================
dataframe_features = dftrain.copy()

target_serie = dataframe_features.pop('sex')

type_feature_dataframe = type_features(dataframe_features)





list_numerical_variables = list(type_feature_dataframe.Feature[type_feature_dataframe.type == 'numerical'])
list_categorical_variables = list(type_feature_dataframe.Feature[type_feature_dataframe.type == 'categorical'])
data_coach = Coach_dataframe(dataframe_features, target_serie, list_numerical_variables, list_categorical_variables)
#%%

data_coach.view_data()
#%%
data_coach.plot_correlation()
data_coach.plot_scatter_matrix()
#%%
data_coach.plot_categorical_var('parch')

#%%
data_coach.scale_dataframe()
#%%
data_coach.treat_linear_correlations(0.8, True)
#%%
steps = 10000
periods = 10
percent_training_data = 80
list_numeric = list_numerical_variables
list_categorical = list_categorical_variables
estimator = 'LinearClassifier'
learning_rate = 1
batch_size = 50
buffer_size = 10000

optimizer = 'Ftrl'
data_coach.train(steps, periods, percent_training_data, list_numeric, list_categorical, estimator, learning_rate,
                 batch_size, buffer_size, optimizer = optimizer )



#%%
##Predictions:
#
feature_dataframe_test = dataframe_features.tail(100)
target_serie_test = target_serie.tail(100)
#%%
how_many = 100
data_coach.plot_predictions_model(feature_dataframe_test, target_serie_test, how_many)
#
#
#%%
data_coach.plot_confusion_matrix(normalize = True, title = 'Model accuracy')

#%%
data_coach.save_figures('DATA')
#dataframe_with_predic = data_coach.dataframe_with_predictions
##%%
#
#plot_confusion_matrix(dataframe_with_predic['Real_values'], dataframe_with_predic['Predictions'],
#                          normalize=False,
#                          title= None,
#                          cmap=plt.cm.Blues)


#%%

data_coach.plot_2categorical_prediction_with_threshold(feature_dataframe_test, target_serie_test, how_many,
                     threshold = 0.1, shuffle_test = False)

#%%
data_coach.plot_confusion_matrix(normalize = True, title = 'Model accuracy')


#%%

data_coach.plot_ROC(feature_dataframe_test, target_serie_test)

#%%
data_coach.evaluate_model(feature_dataframe_test, target_serie_test)
#%%
evaluation_of_the_model = data_coach.evaluation
#%%
data_coach.save_figures('DATA')
