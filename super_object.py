#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:41:48 2019

@author: aramos
"""

from object_functions_tf import *

class Coach_dataframe(training):
    
    def __init__(self, dataframe_features, target_serie, list_numerical_variables, list_categorical_variables):
        
        super().__init__(dataframe_features, target_serie, list_numerical_variables, list_categorical_variables)
        
        self.figure_directory = os.getcwd()
        self.figure_scatter = None
        self.figure_corr = None
        self.figure_categorical_sex = None
        self.figure_training = None
        self.figure_unscale_data = None
        self.figure_predictions = None
        
    def save_figures(self, figure_name):
        print('Saving figures..')
        if self.figure_scatter != None:
            
            self.figure_scatter.savefig(self.figure_directory + '/' + figure_name + '_scatter_.png')   
            print('Scatter matrix: ' + figure_name + '_scatter_.png' +'... Saved')
            
        if self.figure_corr != None:
            
            self.figure_corr.savefig(self.figure_directory + '/' + figure_name + '_Correlation_.png')   
            print('Correlation matrix: ' + figure_name + '_Correlation.png' +'... Saved')   
            

        
        if self.figure_categorical_sex != None:
            
            self.figure_categorical_sex.savefig(self.figure_directory + '/' + figure_name + '_' + self.column_categorical_name + 
                                                '_Categorical_.png')   
            print('Categorical Count: ' + figure_name + '_' + self.column_categorical_name + 
                                                '_Categorical_.png' +'... Saved')   
        
        
        if self.figure_training != None:
            
            self.figure_training.savefig(self.figure_directory + '/' + figure_name + '_Training_.png')   
            print('Training Model: ' + figure_name + '_training.png' +'... Saved')  
            
        if self.figure_unscale_data != None:
            
            self.figure_unscale_data.savefig(self.figure_directory + '/' + figure_name + '_Predictions_Unscaled.png')   
            print('Predictions_Unscaled: ' + figure_name + '_Predictions_Unscaled.png' +'... Saved') 
        
        if self.figure_predictions != None:
            
            self.figure_predictions.savefig(self.figure_directory + '/' + figure_name + '_Predictions.png')   
            print('Predictions: ' + figure_name + '_Predictions.png' +'... Saved')
            
#%%
if __name__ == '__main__':
  #%%  
    #DATAFRAME CREATION
    size_matrix = 270*3
    data = np.random.normal(10, size = size_matrix).reshape(int(size_matrix/3), 3)
    
    dataframe = pd.DataFrame(data = data, columns = ['a', 'b', 'c'])
    dataframe['d'] = 3*dataframe['a'] + np.random.normal(size = dataframe['a'].size)
    dataframe['e'] = 3*dataframe['a'] + np.random.normal(size = dataframe['a'].size)
    dataframe['target_test'] = 4*dataframe['a']*dataframe['b']+ dataframe['c'] + np.random.normal(size = dataframe['a'].size)
    dataframe['categorical_2'] =1.5* dataframe['a']-dataframe['b']
    dataframe['categorical_2'] = np.round(abs(np.array(dataframe['categorical_2']))/np.array(dataframe['categorical_2'].max()))
    mask = dataframe['categorical_2'] == 1
    dataframe['categorical_2'][mask] = 'white'
    mask = dataframe['categorical_2'] == 0
    dataframe['categorical_2'][mask] = 'black'    
    
    target_serie = dataframe.pop('target_test')
#%%   
    dataframe['categorical_1'] = np.round(np.random.normal(loc = 3.2, size = dataframe['a'].size))
    
    
    dataframe['categorical_1'] = np.round(abs(np.array(dataframe['categorical_1']))/np.array(dataframe['categorical_1'].max()))
    mask = dataframe['categorical_1'] == 1
    dataframe['categorical_1'][mask] = 'yes'
    mask = dataframe['categorical_1'] == 0
    dataframe['categorical_1'][mask] = 'no'
    ######################
    
    #Construct the coach_dataframe object

    dataframe_model = Coach_dataframe(dataframe, target_serie, list_numerical_variables =['a', 'b', 'c', 'd', 'e'], list_categorical_variables = ['categorical_1', 'categorical_2'])
    
    name_for_figures = 'FIGURES'
    dataframe_model.plot_scatter_matrix()
    dataframe_model.plot_correlation()
    column_categorical_name = 'categorical_1'
    dataframe_model.plot_categorical_var(column_categorical_name)
    
    try:
        os.mkdir('/home/aramos/Desktop/MachineLearning_Learning/Tensor_flow_1/user_functions/alberto')
        
    except FileExistsError: 
        pass
        
    dataframe_model.figure_directory = '/home/aramos/Desktop/MachineLearning_Learning/Tensor_flow_1/user_functions/alberto'
    dataframe_model.save_figures(name_for_figures)
    
    
    
    name_for_figures = 'Scaled_and_High_correlations_removed_'
    dataframe_model.scale_dataframe()
    dataframe_model.treat_linear_correlations(correlation_threshold = 0.9, remove_from_dataframe = True)
    dataframe_model.plot_scatter_matrix()
    dataframe_model.plot_correlation()
    dataframe_model.save_figures(name_for_figures)

    #Scale target
    dataframe_model.scale_target()

    #training    
    steps = 10000
    periods = 10
    percent_training_data = 80
    list_numeric = ['a', 'b', 'c']
    list_categorical = ['categorical_1', 'categorical_2']  
    estimator = 'LinearRegressor'
#    estimator = 'DNNRegressor'
    learning_rate = 0.5
    batch_size = 5
    buffer_size = 100
    
    
    
    
    bucketize = True
    dict_variables_bucketized = {'a': 4, 'b': 20}                
    list_crossed_features = [['a', 'b']]
    hash_bucket_size = [100] 
    optimizer = 'Ftrl'
    clip_gradient = 5 
    n_classes = None
    hidden_units = [20]
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
    
    dataframe_model.save_figures(name_for_figures)
    
    
#%%
    

    

    target_serie_test = target_serie.head(80)
    feature_dataframe_test = dataframe.head(80)
#%%
    how_many = 30
    dataframe_model.plot_predictions_model(feature_dataframe_test, target_serie_test, 
                                           how_many, shuffle_test = True)    
    

    dataframe_model.save_figures('after_scale')
    
    #%%
# =============================================================================
#     Classification proof
# =============================================================================
    
      #%%  
    #DATAFRAME CREATION
    
    size_matrix = 270*3
    data = np.random.normal(10, size = size_matrix).reshape(int(size_matrix/3), 3)
    
    dataframe = pd.DataFrame(data = data, columns = ['a', 'b', 'c'])
    dataframe['d'] = 3*dataframe['a'] + np.random.normal(size = dataframe['a'].size)
    dataframe['e'] = 3*dataframe['a'] + np.random.normal(size = dataframe['a'].size)
    dataframe['target_test'] = 4*dataframe['a']*dataframe['b']+ dataframe['c'] + np.random.normal(size = dataframe['a'].size)
    dataframe['categorical_2'] =1.5* dataframe['a']-dataframe['b']
    dataframe['categorical_2'] = np.round(abs(np.array(dataframe['categorical_2']))/np.array(dataframe['categorical_2'].max()))
    mask = dataframe['categorical_2'] == 1
    dataframe['categorical_2'][mask] = 'white'
    mask = dataframe['categorical_2'] == 0
    dataframe['categorical_2'][mask] = 'black'    
    
    target_serie = dataframe.pop('categorical_2')

    dataframe['categorical_1'] = np.round(np.random.normal(loc = 3.2, size = dataframe['a'].size))
    
    
    dataframe['categorical_1'] = np.round(abs(np.array(dataframe['categorical_1']))/np.array(dataframe['categorical_1'].max()))
    mask = dataframe['categorical_1'] == 1
    dataframe['categorical_1'][mask] = 'yes'
    mask = dataframe['categorical_1'] == 0
    dataframe['categorical_1'][mask] = 'no'
    ######################
    
    #Construct the coach_dataframe object

    dataframe_model = Coach_dataframe(dataframe, target_serie, list_numerical_variables =['a', 'b', 'c', 'd', 'e'], list_categorical_variables = ['categorical_1'])
    
    name_for_figures = 'FIGURES'
    dataframe_model.plot_scatter_matrix()
    dataframe_model.plot_correlation()
    column_categorical_name = 'categorical_1'
    dataframe_model.plot_categorical_var(column_categorical_name)
    
    try:
        os.mkdir('/home/aramos/Desktop/MachineLearning_Learning/Tensor_flow_1/user_functions/alberto')
        
    except FileExistsError: 
        pass
        
    dataframe_model.figure_directory = '/home/aramos/Desktop/MachineLearning_Learning/Tensor_flow_1/user_functions/alberto'
    dataframe_model.save_figures(name_for_figures)
    
    
    
    name_for_figures = 'Scaled_and_High_correlations_removed_'
    dataframe_model.scale_dataframe()
    dataframe_model.treat_linear_correlations(correlation_threshold = 0.9, remove_from_dataframe = True)
    dataframe_model.plot_scatter_matrix()
    dataframe_model.plot_correlation()
    dataframe_model.save_figures(name_for_figures)

    #Scale target
#    dataframe_model.scale_target()

    #training    
    steps = 10000
    periods = 10
    percent_training_data = 80
    list_numeric = ['a', 'b', 'c']
    list_categorical = ['categorical_1']  
    estimator = 'LinearClassifier'
#    estimator = 'DNNRegressor'
    learning_rate = 0.5
    batch_size = 5
    buffer_size = 100
    
    
    
    
    bucketize = True
    dict_variables_bucketized = {'a': 4, 'b': 20}                
    list_crossed_features = [['a', 'b']]
    hash_bucket_size = [100] 
    optimizer = 'Ftrl'
    clip_gradient = 5 
    n_classes = None
    hidden_units = [20]
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
    
    dataframe_model.save_figures(name_for_figures)
    
    

    

    

    target_serie_test = target_serie.head(80)
    feature_dataframe_test = dataframe.head(80)

    how_many = 30
    dataframe_model.plot_predictions_model(feature_dataframe_test, target_serie_test, 
                                           how_many, shuffle_test = True)    
    

    dataframe_model.save_figures('after_scale')
    
    
    
    
    
    
    
    
    
    
    
    