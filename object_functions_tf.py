#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:13:19 2019

@author: aramos
"""

from functions_for_analysis_dataframes import *






class training(object):
    
    
    def __init__(self, dataframe_features, target_serie, list_numerical_variables, list_categorical_variables):
        
        self.dataframe = dataframe_features.copy()
        self.target_serie = target_serie.copy()
        
        
        #total dataframe
        self.dataframe_target = pd.DataFrame(target_serie, columns = ['target'])

        self.categorical = list_categorical_variables
        self.numerical = list_numerical_variables
        
        self.dataframe_numerical_features = self.dataframe[self.numerical] 
        self.dataframe_categorical_features = self.dataframe[self.categorical]
        
        
    def view_data(self):
        self.total_dataframe = pd.concat([self.dataframe, self.dataframe_target], axis = 1, sort = False)
        dataframe_view(self.total_dataframe)


    def plot_scatter_matrix(self):
        self.total_dataframe = pd.concat([self.dataframe, self.dataframe_target], axis = 1, sort = False)
        self.ax_scatter = scatter_matrix(self.total_dataframe)

        plt.pause(0.1)
    
    def plot_correlation(self):
       self.total_dataframe = pd.concat([self.dataframe, self.dataframe_target], axis = 1, sort = False)
       self.ax_corr = correlation_figure(self.total_dataframe)

       plt.pause(0.1)

    def treat_linear_correlations(self, correlation_threshold, remove_from_dataframe):



        to_drop, record_collinear, self.dataframe = give_variables_linear_correlated(self.dataframe, 
                                 correlation_threshold = correlation_threshold, 
                                 remove_from_dataframe = remove_from_dataframe)


        for drop in to_drop:
            self.numerical.remove(drop)

        return self.dataframe


    def plot_categorical_var(self, column_categorical_name):
        
        plot_categorical_counts(self.dataframe, column_categorical_name)
        
        
        
    def scale_dataframe(self):
        
        self.dataframe_numerical_features = self.dataframe[self.numerical] 
        self.dataframe_categorical_features = self.dataframe[self.categorical]
        
        dataframe_numerical_features = scaling_dataframe(self.dataframe_numerical_features)
        
        self.dataframe = pd.concat([dataframe_numerical_features, self.dataframe_categorical_features], axis = 1, sort = False)
        
        return self.dataframe
        
    

        
    def train(self, steps, periods, percent_training_data, 
                   list_numeric, list_categorical, estimator, learning_rate, batch_size, buffer_size, 
                   bucketize = False, dict_variables_bucketized = {}, 
                  list_crossed_features = None, hash_bucket_size = [], 
                  optimizer = 'GradientDescent',
                   clip_gradient = 5, 
                   n_classes = 2,
                   hidden_units = None,
                  regularization = 'L2',  regularization_strength = None):
        
        
        self.estimator = estimator
        self.model = training_model(steps, periods, self.dataframe, self.target_serie, percent_training_data, 
                       list_numeric, list_categorical, estimator, learning_rate, batch_size, buffer_size, 
                       bucketize, dict_variables_bucketized, 
                      list_crossed_features, hash_bucket_size, 
                      optimizer,
                       clip_gradient, 
                       n_classes,
                       hidden_units,
                      regularization,  regularization_strength)
        
        
        
    
    
    def plot_predictions_model(self, feature_dataframe_test, target_serie_test, target_name, how_many, shuffle_test = False):
        
        
        
        plot_predictions(self.model, self.estimator, feature_dataframe_test, target_serie_test, target_name,
                         how_many, shuffle_test)
        
#%%
if __name__ == '__main__':
    
    
    size_matrix = 270*3
    data = np.random.normal(10, size = size_matrix).reshape(int(size_matrix/3), 3)
    
    dataframe = pd.DataFrame(data = data, columns = ['a', 'b', 'c'])
    dataframe['d'] = 3*dataframe['a'] + np.random.normal(size = dataframe['a'].size)
    dataframe['e'] = 3*dataframe['a'] + np.random.normal(size = dataframe['a'].size)
    dataframe['target'] = 4*dataframe['a']*dataframe['b']+ dataframe['c'] + np.random.normal(size = dataframe['a'].size)
    target_serie = dataframe.pop('target')
    
    dataframe['categorical_1'] = np.round(np.random.normal(loc = 3.2, size = dataframe['a'].size))
    dataframe['categorical_1'] = np.round(abs(np.array(dataframe['categorical_1']))/np.array(dataframe['categorical_1'].max()))
    mask = dataframe['categorical_1'] == 1
    dataframe['categorical_1'][mask] = 'yes'
    mask = dataframe['categorical_1'] == 0
    dataframe['categorical_1'][mask] = 'no'
    
    dataframe_model = training(dataframe, target_serie, list_numerical_variables =['a', 'b', 'c', 'd', 'e'], list_categorical_variables = ['categorical_1'])
    dataframe_model.view_data()
    
    

    dataframe_model.plot_scatter_matrix()
    dataframe_model.plot_correlation()
    
    print('Removing highly linear correlated features...')
    dataframe_model.treat_linear_correlations(correlation_threshold = 0.9, remove_from_dataframe = True)
    
    

    print('\n------------')
    dataframe_model.view_data()
    time.sleep(5)
    dataframe_model.plot_correlation()
    dataframe_model.plot_scatter_matrix()
#    
#    
    print('Plotting categorical variables...')
    dataframe_model.plot_categorical_var('categorical_1')
#    
#    
    print('Scaling dataframe')
    dataframe_model.scale_dataframe()
    dataframe_model.view_data()
    dataframe_model.plot_scatter_matrix()
    dataframe_model.plot_correlation()
    
    #%%
# =============================================================================
#     TRAINING MODEL
# =============================================================================
    steps = 100
    periods = 10
    percent_training_data = 80
    list_numeric = ['a', 'b', 'c']
    list_categorical = ['categorical_1']  
    estimator = 'LinearRegressor'
    learning_rate = 0.005
    batch_size = 100
    buffer_size = 1000
    
    
    
    
    bucketize = True 
    dict_variables_bucketized = {'a': 10, 'b': 20}                
    list_crossed_features = [['a', 'b']]
    hash_bucket_size = [100] 
    optimizer = 'Adagrad'
    clip_gradient = 5 
    n_classes = None
    hidden_units = [10,10]
    regularization = 'L2'  
    regularization_strength = None
    
    dataframe_model.train(steps = steps, periods = periods, percent_training_data = percent_training_data, 
                   list_numeric = list_numeric, list_categorical = list_categorical, estimator = estimator, learning_rate = learning_rate, batch_size = batch_size, buffer_size = buffer_size, 
                   bucketize = bucketize, dict_variables_bucketized = dict_variables_bucketized, 
                  list_crossed_features = list_crossed_features, hash_bucket_size = hash_bucket_size, 
                  optimizer = optimizer,
                   clip_gradient = clip_gradient, 
                   n_classes = n_classes,
                   hidden_units = hidden_units,
                  regularization = regularization,  regularization_strength = regularization_strength)
    
    
    
    
    
    
    
    