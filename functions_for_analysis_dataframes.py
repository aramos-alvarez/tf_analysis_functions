#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:59:55 2019

@author: aramos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:00:02 2019

@author: aramos
"""

import os

#os.chdir('/home/aramos/Desktop/MachineLearning_Learning/Tensor_flow_1/user_functions')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics, preprocessing

from sklearn.utils.multiclass import unique_labels
from IPython import display
import pandas as pd
from tensorflow.python.data import Dataset
import seaborn as sns
import time
from tensorflow.keras import layers
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
    _ = sns.heatmap(correlation, ax = ax_corr, annot = True, cmap = 'PuOr')
    figure_corr.canvas.draw()
    plt.pause(0.01)
    return figure_corr



def give_variables_linear_correlated(dataframe, correlation_threshold = 0.9, remove_from_dataframe = False):
    
    #Retrun a list with variables correlated to remove and a sumary ('record_collinear'), and the dataframe
    
    #ex: to_drop, record_collinear, dataframe = give_variables_linear_correlated(dataframe, correlation_threshold = 0.9, 
    #                                                                        remove_from_dataframe = True)
    
    corr_matrix = dataframe.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
    display.display(to_drop)
    
    
    
    record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])
    
            # Iterate through the columns to drop to record pairs of correlated features
    for column in to_drop:
    
        # Find the correlated features
        corr_features = list(upper.index[upper[column].abs() > correlation_threshold])
    
        # Find the correlated values
        corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
        drop_features = [column for _ in range(len(corr_features))]    
    
        # Record the information (need a temp df for now)
        temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                         'corr_feature': corr_features,
                                         'corr_value': corr_values})
    
        # Add to dataframe
        record_collinear = record_collinear.append(temp_df, ignore_index = True)
    
    
    if remove_from_dataframe:
        for feature in to_drop:
            dataframe.pop(feature)
    
    
    return to_drop, record_collinear, dataframe




def scatter_matrix(dataframe):
    plt.close('Scatter Matrix')
    print('Scatter Matrix')
    figure_scatter = plt.figure('Scatter Matrix', figsize = (10,10))
    ax_scatter = figure_scatter.add_subplot(1,1,1)

    _ = pd.plotting.scatter_matrix(dataframe, ax = ax_scatter)
    figure_scatter.canvas.draw()
    plt.pause(0.01)
    return figure_scatter




def plot_categorical_counts(dftrain, column_categorical_name):
    
    plt.close('Categorical var: ' + column_categorical_name )
    figure_categorical_sex = plt.figure('Categorical var: ' + column_categorical_name )
    ax_categorical_sex = figure_categorical_sex.add_subplot(1,1,1)
    print(dftrain[column_categorical_name].value_counts())
    _ = dftrain[column_categorical_name].value_counts().plot(ax = ax_categorical_sex, kind = 'barh')
    figure_categorical_sex.canvas.draw()
    plt.pause(0.01)
    return figure_categorical_sex




def scaling_serie(serie):
    serie_min = serie.min()
    serie_max = serie.max()
    scale = abs(serie_max - serie_min)
    serie = serie.apply(lambda x: 2*((x - serie_min) / scale - 0.5))
    
    return serie



def inv_scaling_serie(serie, serie_min, serie_max):

    scale = abs(serie_max - serie_min)
    serie = serie.apply(lambda x: (x  /2 + 0.5)*scale + serie_min)
    
    return serie



def scaling_dataframe(dataframe):
    feature_names = list(dataframe.keys())
    
    for feature_name in feature_names:
        dataframe[feature_name] = scaling_serie(dataframe[feature_name])
    
    return dataframe
    



def preprocess_dataframe_simple_way(dataframe, target_name):
    
    dataframe_temp = dataframe.copy()
    target = dataframe_temp.pop(target_name)
    
    return dataframe_temp, target






# prepare the input function
def my_input_fn(feature_dataframe, target_serie, batch_size, num_epochs, shuffle, buffer_size):
    
    
    
    ds = Dataset.from_tensor_slices((dict(feature_dataframe), target_serie))
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    if shuffle:
        ds = ds.shuffle(buffer_size)
    
    
   
    return ds.make_one_shot_iterator().get_next()




def get_boundaries_quantile(serie, num_buckets):
    

    #print('getting_boundaries')
    boundaries = np.arange(0, num_buckets) / num_buckets
    quantile_boundaries = list(serie.quantile(boundaries))
    
    
    
   # print(quantile_boundaries)
    
    #This line is include to avoid problems when you can not boundarized because the number is too big
    #by reducing the number of buckets in steps of -1
    while len(quantile_boundaries) != len(set(quantile_boundaries)):
        num_buckets -= 1
        boundaries = np.arange(0, num_buckets) / num_buckets
        quantile_boundaries = list(serie.quantile(boundaries))
        #print(quantile_boundaries)
        
        
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
            
            if estimator == 'DNNRegressor' or estimator == 'DNNClassifier':
                feature_categorical_embedded = tf.feature_column.embedding_column(feature_categorical, dimension = 
                                                                        int(len(vocabulary)**0.25))
            
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
                #display.display(features_crossed_temporal)
                feature_cro = tf.feature_column.crossed_column(features_crossed_temporal, 
                                                     hash_bucket_size = hash_bucket)
                
                if estimator == 'DNNRegressor' or estimator == 'DNNClassifier':
                    feature_cro = tf.feature_column.embedding_column(feature_cro, dimension = hash_bucket)
                feature_column.append(feature_cro)
        
    else:
        print('NOT CROSSED FEATURES')
        
    return set(feature_column) 



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



def considered_elements(dataframe, list_numeric, list_categorical):
    
    #Function which return a list of numerical and categorical ONLY if this elements are in the dataframe
    if list_categorical == None:
        print('CATEGORICAL LIST EMPTY')
        list_categorical = []
    if list_numeric == None:
        print('NUMERICAL LIST EMPTY')
        list_numeric = []
    
    
    list_elements = list_numeric
    considered_elements_numeric = []
    considered_elements_categorical = []
    
    for element in list_elements:
        if element in dataframe:
            considered_elements_numeric.append(element)
        else:
            display.display(str(element) + ' NOT IN DATAFRAME')
    
        list_elements = list_categorical
    for element in list_elements:
        if element in dataframe:
            considered_elements_categorical.append(element)
        else:
            display.display(str(element) + ' NOT IN DATAFRAME')
            
    if len(considered_elements_numeric) == 0:
        considered_elements_numeric = None
    if len(considered_elements_categorical) == 0:
        considered_elements_categorical = None
    return considered_elements_numeric, considered_elements_categorical


def model_creator(dataframe, list_numeric, bucketize, dict_variables_bucketized, 
                  list_categorical, list_crossed_features, hash_bucket_size, 
                  estimator, learning_rate, clip_gradient, optimizer = 'GradientDescent', hidden_units = None,
                  regularization = 'L2',  regularization_strength = None, n_classes = 2):
    
    #estimator should be: LinearRegressor, LinearClassifier, DNNRegressor (in the case of Neural Network)
    my_optimizer = optimizer_creator(learning_rate, clip_gradient, optimizer, regularization, 
                     regularization_strength)
    
    
    
    #chekc if all elements in list numeric and list_categorical are in the dataframe
    list_numeric, list_categorical = considered_elements(dataframe, list_numeric, list_categorical)
    
    print('\n..............Trying to construct feature with .........\n')
    print('Numeric \n---------')
    display.display(list_numeric)
    print('Categorical \n---------')
    display.display(list_categorical)
    print('----------------------------')
    feature_columns = construct_feature_columns(dataframe, estimator, 
                                                list_numeric = list_numeric, bucketize = bucketize, dict_variables_bucketized = dict_variables_bucketized, 
                              list_categorical = list_categorical, list_crossed_features = list_crossed_features,
                                                hash_bucket_size = hash_bucket_size)

    
    
    display.display(str(feature_columns))
    if estimator == 'LinearRegressor':
        
        print('\n Estimator:   ' + estimator)
        model = tf.estimator.LinearRegressor(feature_columns = feature_columns, optimizer = my_optimizer)
        
        
    elif estimator == 'LinearClassifier':
        print('\n Estimator:   ' + estimator)
        model = tf.estimator.LinearClassifier(feature_columns = feature_columns, n_classes = n_classes,
                                              
                                              optimizer = my_optimizer)
        
        
    elif estimator == 'DNNRegressor':
        print('\n Estimator:   ' + estimator + ' with hidden_units:...' + str(hidden_units))
        model = tf.estimator.DNNRegressor(feature_columns = feature_columns, hidden_units = hidden_units , 
                                          optimizer = my_optimizer)
        
        
        
    elif estimator == 'DNNClassifier':
        
        print('\n Estimator:   ' + estimator + ' with hidden_units:...' + str(hidden_units))
        model = tf.estimator.DNNClassifier(feature_columns = feature_columns,  n_classes = n_classes,
                                           hidden_units = hidden_units, 
                                          optimizer = my_optimizer,
                                          config = tf.contrib.learn.RunConfig(keep_checkpoint_max = 1))
        
        
    
    else:
        print('\n BAD DEFINED ESTIMATOR, use: LinearRegressor, LinearClassifier, DNNRegressor or DNNClassifier')
    
    

    return model, feature_columns



def training_model(steps, periods, feature_dataframe, target_serie, percent_training_data, 
                   list_numeric, list_categorical, estimator, learning_rate, batch_size, buffer_size, 
                   bucketize = False, dict_variables_bucketized = {}, 
                  list_crossed_features = None, hash_bucket_size = [], 
                  optimizer = 'GradientDescent',
                   clip_gradient = 5, 
                   n_classes = 2,
                   hidden_units = None,
                  regularization = 'L2',  regularization_strength = None):


    #steps---> number of times that the model will be trained
    #periods--> how many times the model will be evaluate during the running (this is not a hyperparameter
    #           its is only a parameter to see the graph of training)
    # n_classes is used in classification problems
    
    steps_per_period = steps / periods
    
    
    total_raws = feature_dataframe.count()[0]
    
    #DATAFRAME FOR TRAINING
    feature_dataframe_training = feature_dataframe.head(int(percent_training_data/ 100 * total_raws))
    target_serie_training = target_serie.head(int(percent_training_data / 100 * total_raws))
    
    #DATAFRAME FOR VALIDATION
    feature_dataframe_validation = feature_dataframe.tail(int(1-(percent_training_data/ 100) * total_raws))
    target_serie_validation = target_serie.tail(int(1-(percent_training_data/ 100) * total_raws))
    
    
    
    #train input fn
    num_epochs = None
    shuffle = True

    train_input = lambda: my_input_fn(feature_dataframe_training , target_serie_training,
                              batch_size, num_epochs, shuffle, buffer_size)
    
    #predict training input fn
    num_epochs = 1
    shuffle = False
    batch_size = 1
    
    predict_training_input = lambda: my_input_fn(feature_dataframe_training , target_serie_training, 
                                         batch_size, num_epochs,shuffle, buffer_size)
    
    
    
    #predict validation input fn
    predict_validation_input = lambda: my_input_fn(feature_dataframe_validation , target_serie_validation, 
                                         batch_size, num_epochs,shuffle, buffer_size)
    
    
    #MODEL CREATION
    
    
    model, feature_columns = model_creator(feature_dataframe_training , list_numeric, bucketize, dict_variables_bucketized, 
                  list_categorical, list_crossed_features, hash_bucket_size, 
                  estimator, learning_rate, clip_gradient, optimizer, hidden_units,
                  regularization ,  regularization_strength )
    
    
    #FIGURE CONSTRUCTOR FOR THE TRAINING DATA
    plt.close('TRAINING MODEL')
    figure_training = plt.figure('TRAINING MODEL', figsize = (8,8))
    ax_training = figure_training.add_subplot(1,1,1)
    
    rmse_training_list = []
    rmse_validation_list = []
    
    log_loss_training_list = []
    log_loss_validation_list = []
    #TRAINING THE MODEL BY STEPS
    print('TRAINING MODEL:...\n' + 'target:...' + target_serie.name)
    if estimator == 'LinearRegressor' or estimator == 'DNNRegressor':
        print('Minimize Root Mean Squared Error \n')
    elif estimator == 'LinearClassifier' or estimator == 'DNNClassifier':
        print('Minimize Log Loss function (clasification problem) \n')
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(target_serie_training)
        
    else:
        print('ERROR defining estimator')
    
    
    for period in range(0, periods):
        
        
        
        _ = model.train(input_fn = train_input, steps = steps_per_period)
        
        #take a break to predict values and calculate the error
        prediction_training = model.predict(input_fn = predict_training_input)
        prediction_validation = model.predict(input_fn = predict_validation_input)
        
        
        #Now depending of the estimator it is neccessary see some items into predictions
        
        if estimator == 'LinearRegressor' or estimator == 'DNNRegressor':
            
            
            name_ylabel = 'Root Mean Squared Error'
            prediction_training = np.array([item['predictions'][0] for item in prediction_training])
            prediction_validation = np.array([item['predictions'][0] for item in prediction_validation])
            
            
            rmse_training = np.sqrt(abs(metrics.mean_squared_error(target_serie_training, 
                                                                   prediction_training)))
            
            
            
            
            print('PERIOD ' + str(period) + '...rmse_training: ' + str(np.round(rmse_training, 2)))
                  
            rmse_training_list.append(rmse_training)
            
            rmse_validation = np.sqrt(abs(metrics.mean_squared_error(target_serie_validation, 
                                                                   prediction_validation)))
            rmse_validation_list.append(rmse_validation)
            
            
            
            #plotting
            ax_training.clear()
            ax_training.plot(rmse_training_list, '-o', color = 'blue', label = 'Training data')
            ax_training.plot(rmse_validation_list, '-x', color = 'red', label = 'Validation data')
            
            figure_training.canvas.draw()
            plt.pause(0.01)
            
            
        
        elif estimator == 'LinearClassifier' or estimator == 'DNNClassifier':
            
            

            
            name_ylabel = 'Log Loss'
            
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
            
            
            
        else:
            print('NOT WELL ESTIMATOR WAS DEFINED!')
        
        
    ax_training.legend()        
    ax_training.grid(linestyle = '--', alpha = 0.3)
    ax_training.text((ax_training.get_xlim()[0]+ax_training.get_xlim()[1])/2, 
                     (ax_training.get_ylim()[0]+ax_training.get_ylim()[1])/1.7, 'Target: ' + target_serie.name , size = 15, color = 'darkred')
    
    
    inform_numeric = str(list_numeric) 
    inform_categorical = str(list_categorical)
    inform_crosses = str(list_crossed_features)
    ax_training.text((ax_training.get_xlim()[0]+ax_training.get_xlim()[1])/2, 
                     (ax_training.get_ylim()[0]+ax_training.get_ylim()[1])/2, 'Numeric Features: ' + inform_numeric, size = 8, color = 'darkgreen')
    
    ax_training.text((ax_training.get_xlim()[0]+ax_training.get_xlim()[1])/2, 
                     (ax_training.get_ylim()[0]+ax_training.get_ylim()[1])/2.05, 'Categorical Features: ' + inform_categorical, size = 8, color = 'darkblue')
    
    
    ax_training.text((ax_training.get_xlim()[0]+ax_training.get_xlim()[1])/2, 
                     (ax_training.get_ylim()[0]+ax_training.get_ylim()[1])/2.1, 'Crosses Features: ' + inform_crosses, size = 8, color = 'violet')
    
    ax_training.set_title('Estimator:' + estimator)
    ax_training.set_xlabel('Period (a.u.)')
    ax_training.set_ylabel(name_ylabel)
    
    print('MODEL TRAINED!!')
    return model, figure_training


# =============================================================================
# Plotting predictions
# =============================================================================
def plot_predictions(model, estimator, feature_dataframe_test, target_serie_test, target_name, how_many, shuffle_test = False, scaled = False):
    
    print('In the case of classification problem the prediction will be this one with higher probability!')
    #this function will return a dataframe with the predictions and the real values
    #IMPORTANT: In the case of classification problem the prediction will be this one with higher probability!
    

    print('Performing prediction with the model: \n-------')
    
    if scaled:
        
        print('Plot with data target scaled \n' )
        
        
        
    feature_dataframe_test = feature_dataframe_test.head(how_many)
    target_serie_test = target_serie_test.head(how_many)
    
    if shuffle_test:
        feature_dataframe_test = feature_dataframe_test.reindex(np.random.permutation(feature_dataframe_test.index))
        target_serie_test = target_serie_test.reindex(np.random.permutation(target_serie_test.index))
    
    
    dataframe_with_predict = pd.DataFrame()
    dataframe_with_predict['Real_values'] = np.array(target_serie_test)
    
    
    num_epochs = 1
    shuffle = False
    batch_size = 1
    buffer_size = 1000
    predict_test_input = lambda: my_input_fn(feature_dataframe_test , target_serie_test, 
                                     batch_size, num_epochs,shuffle, buffer_size)
    
    print('Performing prediction.........')
    predictions_test = model.predict(input_fn = predict_test_input)
    
    if estimator == 'LinearRegressor' or estimator == 'DNNRegressor':
        
        print(estimator)
        predictions_test = np.array([item['predictions'][0] for item in predictions_test])
        
       
        
        dataframe_with_predict['Predictions'] = predictions_test
        print('Creating Figure object for testing...')
        plt.close('Predictions Vs Actual values')
        figure_prediction = plt.figure('Predictions Vs Actual values')
        ax_prediction = figure_prediction.add_subplot(1,1,1)
        
        ax_prediction.plot(np.array(target_serie_test), 'o-', label = 'Actual values')
        ax_prediction.plot(predictions_test, 'x-', label = 'Predictions')
        ax_prediction.legend()
        
        figure_prediction.canvas.draw()
        plt.pause(0.01)
        
        if scaled:
            ax_prediction.set_title('Data Scaled')
        ax_return = ax_prediction    
        figure_return = figure_prediction
        
        dataframe_with_predict['Predictions'] = predictions_test
    elif estimator == 'LinearClassifier' or estimator == 'DNNClassifier':
        
        dataframe_counts_categorical = pd.DataFrame()
        
        list_target_test = list(target_serie_test)
        #Numerical output
        predictions_test = [item['class_ids'][0] for item in predictions_test]
        
        dataframe_with_predict['Predictions'] = np.array(predictions_test)
        vocabulary = target_serie_test.unique()
        
        dataframe_counts_categorical['list_categorical'] = vocabulary
        count_real = []
        count_predictions = []
        
        
        for item in vocabulary:
            
            count_real.append(list_target_test.count(item))
##            
            count_predictions.append(predictions_test.count(item))
#        
        
        dataframe_counts_categorical['count_real'] = np.array(count_real)
        dataframe_counts_categorical['count_predictions'] = np.array(count_predictions)
        plt.close('Bar count')
        figure_bar_count = plt.figure('Bar count')
        ax_figure = figure_bar_count.add_subplot(1,1,1)
        ax_bar = dataframe_counts_categorical.plot.barh('list_categorical', 'count_real', ax = ax_figure)
        ax_bar = dataframe_counts_categorical.plot.barh('list_categorical', 'count_predictions', ax = ax_bar, 
                                                        color = 'red', alpha = 0.2)
        
        ax_bar.set_ylabel(target_name)
        ax_bar.grid(color = 'green', linestyle='--')
        figure_bar_count.canvas.draw()
        plt.pause(0.01)
        
        ax_return = ax_bar
        figure_return = figure_bar_count
        
    else:
        print('Bad estimator defined')

    return dataframe_with_predict, figure_return



def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
#    Ex.
#    target = ['a', 'b', 'c', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']
#
#    prediction = ['a', 'b', 'c', 'b', 'a', 'a', 'a', 'a', 'a', 'a', 'c']
#    plot_confusion_matrix(target, prediction, normalize = True, title = 'Normalized', cmap = plt.cm.Oranges)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = list(unique_labels(y_true, y_pred))
    display.display(classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.close('Confusion matrix')
    fig = plt.figure('Confusion matrix')
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
#
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.1)
    return ax



def plot_ROC_curve(model, feature_dataframe_test, target_serie_test):
       
    #This is done for binary classification
    num_epochs = 1
    shuffle = False
    batch_size = 1
    buffer_size = 1000
    prediction_eval_input = lambda: my_input_fn(feature_dataframe_test , target_serie_test, 
                                     batch_size, num_epochs,shuffle, buffer_size)    

    prediction_eval = model_Titanic.predict(prediction_eval_input)
    positive_score = np.array([item['probabilities'][1] for item in prediction_eval])
    
    fpr, tpr, threshold = metrics.roc_curve(target_serie_test, positive_score)
    
    plt.close('ROC curve')
    figure_roc = plt.figure('ROC curve')
    ax_roc = figure_roc.add_subplot(1,1,1)
    ax_roc.plot(fpr, tpr)
    ax_roc.plot([0, 1], [0, 1])
    ax_roc.grid()
    figure_roc.canvas.draw()
    plt.pause(0.1)
    return ax_roc





    
    
def evaluation_model(model, feature_eval, target_eval):

    shuffle = False
    num_epochs = 1
    batch_size = 1
    buffer_shuffle = None
    prediction_eval_input = lambda: my_input_fn(feature_eval, target_eval, batch_size, 
                                                num_epochs, shuffle, buffer_shuffle)
    
    evaluation = model.evaluate(prediction_eval_input)
    display.display(evaluation)
    return evaluation












