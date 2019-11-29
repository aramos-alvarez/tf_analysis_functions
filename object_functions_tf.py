#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:13:19 2019

@author: aramos
"""

from functions_for_analysis_dataframes import *






class training(object):
    
    "Comments about the value of this object"
    def __init__(self, dataframe_features, target_serie, list_numerical_variables, list_categorical_variables):
        
        self.dataframe = dataframe_features.copy()
        self.target_serie = target_serie.copy()
        
        
        #total dataframe
        self.dataframe_target = pd.DataFrame(target_serie, columns = [target_serie.name])

        self.categorical = list_categorical_variables
        self.numerical = list_numerical_variables
        
        self.dataframe_numerical_features = self.dataframe[self.numerical] 
        self.dataframe_categorical_features = self.dataframe[self.categorical]
        self.scale = False
        self.scale_target_serie = False
        
    def view_data(self):
        self.total_dataframe = pd.concat([self.dataframe, self.dataframe_target], axis = 1, sort = False)
        dataframe_view(self.total_dataframe)


    def plot_scatter_matrix(self):
        self.total_dataframe = pd.concat([self.dataframe, self.dataframe_target], axis = 1, sort = False)
        self.figure_scatter = scatter_matrix(self.total_dataframe)

        plt.pause(0.1)
    
    def plot_correlation(self):
       self.total_dataframe = pd.concat([self.dataframe, self.dataframe_target], axis = 1, sort = False)
       self.figure_corr = correlation_figure(self.total_dataframe)

       plt.pause(0.1)

    def treat_linear_correlations(self, correlation_threshold, remove_from_dataframe):



        to_drop, record_collinear, self.dataframe = give_variables_linear_correlated(self.dataframe, 
                                 correlation_threshold = correlation_threshold, 
                                 remove_from_dataframe = remove_from_dataframe)

        
        self.numerical_past = self.numerical.copy()
        for drop in to_drop:
            self.numerical.remove(drop)

        return self.dataframe


    def plot_categorical_var(self, column_categorical_name):
        
        self.figure_categorical_sex = plot_categorical_counts(self.dataframe, column_categorical_name)
        self.column_categorical_name = column_categorical_name
        
     
    def recovery_old_dataframe_and_target(self):
        self.dataframe = self.dataframe_past.copy()
        self.numerical = self.numerical_past
        self.target_serie = self.target_serie_past.copy()
        self.scale = False
        self.scale_target_serie = False
    
    def scale_target(self):
        
        self.scale_target_serie = True
        self.target_serie_past = self.target_serie.copy()
        self.target_serie_max = self.target_serie.max()
        self.target_serie_min = self.target_serie.min()
        self.target_serie = scaling_serie(self.target_serie)
    
    
    
    def scale_dataframe(self):
        
        self.scale = True
        self.dataframe_past = self.dataframe.copy()
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
        self.model, self.figure_training = training_model(steps, periods, self.dataframe, self.target_serie, percent_training_data, 
                       list_numeric, list_categorical, estimator, learning_rate, batch_size, buffer_size, 
                       bucketize, dict_variables_bucketized, 
                      list_crossed_features, hash_bucket_size, 
                      optimizer,
                       clip_gradient, 
                       n_classes,
                       hidden_units,
                      regularization,  regularization_strength)
        
        
        
    
    
    def plot_predictions_model(self, feature_dataframe_test, target_serie_test, how_many, shuffle_test = False):
        
        
        if self.scale:
            
            #If I have feed the model with the dataframe scaled I should use the same way with the dataframe test
            
            #Scale feature_dataframe_test
            feature_dataframe_test_numerical_features =   feature_dataframe_test[self.numerical]
            feature_dataframe_test_categorical_features = feature_dataframe_test[self.categorical]
            feature_dataframe_test_numerical_features = scaling_dataframe(feature_dataframe_test_numerical_features)
            feature_dataframe_test = pd.concat([feature_dataframe_test_numerical_features, feature_dataframe_test_categorical_features], 
                                               axis = 1, sort = False)
       
        
        
            
        if self.scale_target_serie:
            #If the model have scaled the serie, I should feed the model with this one scaled, with the parameters to scale before

            #Scale target_serie_test
            scale = abs(self.target_serie_max - self.target_serie_min)
            target_serie_test = target_serie_test.apply(lambda x: 2*((x - self.target_serie_min) / scale - 0.5))
            
            
            
        self.dataframe_with_predictions, self.figure_predictions = plot_predictions(self.model, self.estimator, feature_dataframe_test, 
                                                                 target_serie_test, target_serie_test.name,
                                                                    how_many, shuffle_test, self.scale_target_serie )
        
        if self.scale_target_serie:
            
            plt.close('Predictions')
            self.figure_unscale_data = plt.figure('Predictions')
            ax_unscale_data = self.figure_unscale_data.add_subplot(1,1,1)
            
            self.dataframe_with_predictions_unscale = self.dataframe_with_predictions.apply(lambda x: (x + 0.5) *scale/2+self.target_serie_min)
        #plot target serie test and Prediction from dataframe_with_predictions
            #I should convert the dataframe with predict in the real values before plot 
            ax_unscale_data.plot(np.array(self.dataframe_with_predictions_unscale['Real_values']), 'o-', label = 'Actual values')
            ax_unscale_data.plot(np.array(self.dataframe_with_predictions_unscale['Predictions']), 'x-', label = 'Predictions')
            ax_unscale_data.legend()
            self.figure_unscale_data.canvas.draw()
            plt.pause(0.01)
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
    steps = 1000
    periods = 10
    percent_training_data = 80
    list_numeric = ['a', 'b', 'c']
    list_categorical = ['categorical_1']  
    estimator = 'DNNRegressor'
    estimator = 'LinearRegressor'
    learning_rate = 20
    batch_size = 10
    buffer_size = 1000
    
    
    
    
    bucketize = True
    dict_variables_bucketized = {'a': 4, 'b': 20}                
    list_crossed_features = [['a', 'b']]
    hash_bucket_size = [100] 
    optimizer = 'Ftrl'
    clip_gradient = 5 
    n_classes = None
    hidden_units = [2,4, 10]
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
    
    
    
    
    
    
    
    