#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:13:19 2019

@author: aramos
"""

from functions_for_analysis_dataframes import *






class training(object):
    
    "Comments about the value of this object"
    
    "type_features(dataframe) is a good function point to determine the list of numerical and categorical variables"
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
        self.encoding_target = False
        self.evaluation ='Not evaluate'
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
        
        #In the case of categorical variable in the target I should encoder that one before train
        
        if self.target_serie.dtype == 'O':
            print('Target categorical object---> Encoding Target...')
            
            self.encoding_target = True
            
            target_name = self.target_serie.name
            
            self.label_encoder = preprocessing.LabelEncoder()
            self.label_encoder.fit(self.target_serie)
            
            self.target_serie = self.label_encoder.transform(self.target_serie)
            self.target_serie = pd.Series(self.target_serie)
            self.target_serie.name = target_name
        
        
        
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
        
        
        
        if self.encoding_target:
            print('The target is categorical:\n Decoding the target serie training...')
            #Uncoding the target
            self.target_serie = self.label_encoder.inverse_transform(self.target_serie)
            
            
    def plot_2categorical_prediction_with_threshold(self, feature_dataframe_test, target_serie_test, how_many,
                     threshold = 0.5, shuffle_test = False):
    
        
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
        
        
        
        self.dataframe_with_predictions, self.figure_predictions_thr = plot_prediction_with_threshold(self.model, self.estimator, 
                                                            feature_dataframe_test, target_serie_test, target_serie_test.name, how_many,
                                                            threshold, shuffle_test, self.scale_target_serie)
        
        
        if self.encoding_target:
            
            self.figure_predictions_thr.clear()
            self.dataframe_with_predictions['Predictions'] = self.label_encoder.inverse_transform(self.dataframe_with_predictions['Predictions'])
            
            
            dataframe_counts_categorical = pd.DataFrame()
        
            list_target_test = list(self.dataframe_with_predictions['Real_values'])
            list_prediction_test = list(self.dataframe_with_predictions['Predictions'])
            vocabulary = target_serie_test.unique()
            dataframe_counts_categorical['list_categorical'] = vocabulary
            count_real = []
            count_predictions = []
        
        
            for item in vocabulary:
                
                count_real.append(list_target_test.count(item))
    ##            
                count_predictions.append(list_prediction_test.count(item))
#        
        
            dataframe_counts_categorical['count_real'] = np.array(count_real)
            dataframe_counts_categorical['count_predictions'] = np.array(count_predictions)
            self.dataframe_counts_categorical = dataframe_counts_categorical
            
            ax_figure = self.figure_predictions_thr.add_subplot(1,1,1)
            ax_figure.set_title('Probability threshold: ' + str(threshold))
            ax_bar = dataframe_counts_categorical.plot.barh('list_categorical', 'count_real', ax = ax_figure, edgecolor='black',)
            ax_bar = dataframe_counts_categorical.plot.barh('list_categorical', 'count_predictions', ax = ax_bar, 
                                                        color = 'red', alpha = 0.2,  hatch="/")
        
            ax_bar.set_ylabel(target_serie_test.name)
            ax_bar.grid(color = 'green', linestyle='--')
            self.figure_predictions_thr.canvas.draw()
            plt.pause(0.01)
                #I should clear the figure and decoding target
        

        
            
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
            
            
           

            #I should encode the target with the same encoder before feed the model
            
            
        self.dataframe_with_predictions, self.figure_predictions = plot_predictions(self.model, self.estimator, feature_dataframe_test, 
                                                                 target_serie_test, target_serie_test.name,
                                                                    how_many, shuffle_test, self.scale_target_serie )
        
        
        if self.encoding_target:
            
            self.figure_predictions.clear()
            self.dataframe_with_predictions['Predictions'] = self.label_encoder.inverse_transform(self.dataframe_with_predictions['Predictions'])
            
            
            dataframe_counts_categorical = pd.DataFrame()
        
            list_target_test = list(self.dataframe_with_predictions['Real_values'])
            list_prediction_test = list(self.dataframe_with_predictions['Predictions'])
            vocabulary = target_serie_test.unique()
            dataframe_counts_categorical['list_categorical'] = vocabulary
            count_real = []
            count_predictions = []
        
        
            for item in vocabulary:
                
                count_real.append(list_target_test.count(item))
    ##            
                count_predictions.append(list_prediction_test.count(item))
#        
        
            dataframe_counts_categorical['count_real'] = np.array(count_real)
            dataframe_counts_categorical['count_predictions'] = np.array(count_predictions)
            self.dataframe_counts_categorical = dataframe_counts_categorical
            
            ax_figure = self.figure_predictions.add_subplot(1,1,1)
            ax_bar = dataframe_counts_categorical.plot.barh('list_categorical', 'count_real', ax = ax_figure, edgecolor='black',)
            ax_bar = dataframe_counts_categorical.plot.barh('list_categorical', 'count_predictions', ax = ax_bar, 
                                                        color = 'red', alpha = 0.2,  hatch="/")
        
            ax_bar.set_ylabel(target_serie_test.name)
            ax_bar.grid(color = 'green', linestyle='--')
            self.figure_predictions.canvas.draw()
            plt.pause(0.01)
                #I should clear the figure and decoding target
        
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
            
    def plot_confusion_matrix(self, normalize = False, title = None, cmap = plt.cm.Blues):
        
        
        if self.dataframe_with_predictions is not None:
            self.confusion_matrix, self.fig_confusion = plot_confusion_matrix(self.dataframe_with_predictions['Real_values'], 
                                                                              self.dataframe_with_predictions['Predictions'],
                                                                                          normalize=normalize,
                                                                                          title= title,
                                                                                          cmap=cmap)
        else:
            print('error plotting confusion matrix---- First execute: plot_predictions_model(...)')
        
    
    def plot_ROC(self, feature_dataframe_test, target_serie_test):
        
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
        
        
        if self.encoding_target:
            print('Target categorical object---> Encoding Target...')
            target_serie_test = self.label_encoder.transform(target_serie_test)
            target_serie_test = pd.Series(target_serie_test)
            
            

        
        
        
        
        self.figure_roc, self.fpr_roc, self.tpr_roc, self.threshold_roc = plot_ROC_curve(self.model, feature_dataframe_test, 
                                                                                         target_serie_test)
    def evaluate_model(self, feature_dataframe_test, target_serie_test):
        feature_dataframe_test = feature_dataframe_test.copy()
        
        if self.scale:
            
            #If I have feed the model with the dataframe scaled I should use the same way with the dataframe test
            
            #Scale feature_dataframe_test
            feature_dataframe_test_numerical_features =   feature_dataframe_test[self.numerical]
            feature_dataframe_test_categorical_features = feature_dataframe_test[self.categorical]
            feature_dataframe_test_numerical_features = scaling_dataframe(feature_dataframe_test_numerical_features)
            feature_dataframe_test = pd.concat([feature_dataframe_test_numerical_features, feature_dataframe_test_categorical_features], 
                                               axis = 1, sort = False)
            print('feature scaled')
        
        if self.scale_target_serie:
            #If the model have scaled the serie, I should feed the model with this one scaled, with the parameters to scale before

            #Scale target_serie_test
            scale = abs(self.target_serie_max - self.target_serie_min)
            target_serie_test = target_serie_test.apply(lambda x: 2*((x - self.target_serie_min) / scale - 0.5))
        
        
        if self.encoding_target:
            print('Target categorical object---> Encoding Target...')
            target_serie_test = self.label_encoder.transform(target_serie_test)
            target_serie_test = pd.Series(target_serie_test)
        
        self.evaluation = evaluation_model(self.model, feature_dataframe_test, target_serie_test)
        
        
    
#%%
if __name__ == '__main__':
    
    
    pass
    
    
    
    