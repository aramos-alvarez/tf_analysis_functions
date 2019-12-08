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
        
        try:
            os.mkdir('figures_saved')
        except FileExistsError:
            pass
        
        self.figure_directory = self.figure_directory + '/figures_saved'
        self.figure_scatter = None
        self.figure_corr = None
        self.figure_categorical_sex = None
        self.figure_training = None
        self.figure_unscale_data = None
        self.figure_predictions = None
        self.fig_confusion = None
        self.figure_predictions_thr = None
        self.figure_roc = None
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
            
        if self.fig_confusion != None:
            self.fig_confusion.savefig(self.figure_directory + '/' + figure_name + '_Confusion_matrix.png')   
            print('Confusion matrix: ' + figure_name + '_Confusion_matrix.png' +'... Saved')
            
        if self.figure_predictions_thr !=None:
            
            self.figure_predictions_thr.savefig(self.figure_directory + '/' + figure_name  +'_' +
                                                self.figure_predictions_thr.get_axes()[0].get_title() +'.png')
            
            print('Classification with threshold: ' + figure_name  +'_' +
                                                self.figure_predictions_thr.get_axes()[0].get_title() +'.png'+'... Saved')
            
        if self.figure_roc != None:
            
            self.figure_roc.savefig(self.figure_directory + '/' + figure_name  +'_ROC_curve.png'  )
            print('ROC curve: '+ figure_name  +'_ROC_curve.png' + '... Saved')
#%%
if __name__ == '__main__':
  #%%  
    #DATAFRAME CREATION
    pass
    
    
    
    
    
    
    
    
    
    
    