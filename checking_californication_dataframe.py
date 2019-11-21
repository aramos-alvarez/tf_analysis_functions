#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:50:17 2019

@author: aramos
"""

from object_functions_tf import *
#%%
print('Donwloading dataframe.....')
california_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

print('Shuffle input data...')
california_dataframe = california_dataframe.reindex(np.random.permutation(california_dataframe.index))
#%%
dataframe = california_dataframe.copy()

list_numerical_features =  list(dataframe.keys())
list_categorical_features = None
target_name = 'median_house_value'

feature_dataframe, target_serie = preprocess_dataframe_simple_way(dataframe, target_name)
list_numerical_features =  list(feature_dataframe.keys())
list_categorical_features = []

#%%
dataframe_model = training(feature_dataframe, target_serie / target_serie.max(), list_numerical_variables =list_numerical_features,
                           list_categorical_variables = list_categorical_features)
dataframe_model.view_data()



dataframe_model.plot_scatter_matrix()
dataframe_model.plot_correlation()

print('Removing highly linear correlated features...')
dataframe_model.treat_linear_correlations(correlation_threshold = 0.85, remove_from_dataframe = True)

time.sleep(2)
dataframe_model.plot_scatter_matrix()
dataframe_model.plot_correlation()

#%%

print('Scaling dataframe')
dataframe_model.scale_dataframe()
dataframe_model.view_data()
dataframe_model.plot_scatter_matrix()
dataframe_model.plot_correlation()

#%%


steps = 5000
periods = 10
percent_training_data = 80
list_numeric = ['longitude', 'housing_median_age', 'total_rooms', 'median_income']
list_categorical = []  
estimator = 'LinearRegressor'
learning_rate = 2
batch_size =  20
buffer_size = 10000




bucketize = True 
dict_variables_bucketized = {'longitude': 10, 'median_income': 20}                
list_crossed_features = [['longitude', 'median_income']]
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


#%%
dataframe_features = california_dataframe.copy()
target_name = 'median_house_value'
target_serie_test = dataframe_features.pop(target_name) 
target_serie_test = target_serie_test.tail(200)
feature_dataframe_test = scaling_dataframe(dataframe_features.tail(200))

target_serie_test =target_serie_test / target_serie_test.max()
how_many = 200
dataframe_model.plot_predictions_model(feature_dataframe_test, target_serie_test, target_name, how_many, shuffle_test = True)


#%%
model = dataframe_model.model
estimator = 'LinearRegressor'
#plot_predictions(model, estimator, feature_dataframe_test, target_serie_test, target_name, how_many, shuffle_test = False)



num_epochs = 1
shuffle = False
batch_size = 1
buffer_size = 1000
predict_test_input = lambda: my_input_fn(feature_dataframe_test , target_serie_test, 
                                 batch_size, num_epochs,shuffle, buffer_size)


predictions_test = model.predict(input_fn = predict_test_input)
predictions_test = np.array([item['predictions'][0] for item in predictions_test])

#%%
plt.close('Predictions Vs Actual values')
figure_prediction = plt.figure('Predictions Vs Actual values')
ax_prediction = figure_prediction.add_subplot(1,1,1)

ax_prediction.plot(np.array(target_serie_test), 'o-', label = 'Actual values')
ax_prediction.plot(predictions_test, 'x-', label = 'Predictions')
ax_prediction.legend()

figure_prediction.canvas.draw()















