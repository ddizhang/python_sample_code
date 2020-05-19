import pandas as pd
import numpy as np
import os
import pickle
import logging
import random
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger(__name__)



def pred_interval_bs(model_obj, model_class, data_fit, new_data, bs_iteration = 500, 
                     retrain_hyper_params = False, full_return = False):
    '''
    Args:
        model_obj: self-defined model trained in main space. With elements:
            self.target: a string indicating the column name of response variable in data_fit and new_data
            self.num_lag_term: number of lag terms
        model_class: the class constructor of model_obj. Can be derived with following code:
                    #         class_name = model_obj.__class__.__name__
                    #         model_bs = eval(class_name)()
                    #         constructor = globals()[class_name]
                    #         model_bs = constructor()

        data_fit: training data, a pandas.DataFrame
        new_data: data to predict, a pandas.DataFrame
        bs_iteration: number of bootstrap iterations

    Returns:
        y_pred: point prediction using main model
        error_matrix: a matrix[i][j] of the prediction error 
                      in each bootstrap iteration (dimension i), for each observation (dimension j).  
    '''
    
    target = model_obj.target
    if type(target) == list:
        target_str = target[0]
        
    num_lag_terms = model_obj.max_num_lag_terms
    pred_period = new_data.shape[0]
    # error matrix[i][j]: i is # bs iteration, j is # terms predicted with the model
    error_matrix = [[None for j in range(pred_period)] for i in range(bs_iteration)] 

    ### Step 1: in Actual Space
    # Step 1.1: fit model from term 1:t
    y_fitted = model_obj.fit(data_fit) #fit_TPA(model_obj, data_fit) 

    ### Step 1.2: centualized fitting error of term 1:t 
    y_actual = data_fit[target].iloc[num_lag_terms:,0].reset_index(drop = True)
    r = y_actual - y_fitted
    r = r - np.mean(r)

    ### Step 1.3: point prediction for t:t+n with original model (not used in bootstrap process)
    y_pred = model_obj.predict(new_data) #predict_TPA(model_obj, new_data)
    
    all_model_bs = []
    
    for k in range(bs_iteration):

        ### Step 2.1: Construct Bootstrap-Space Actuals on term 1:t
        y_actual_bs = model_obj.predict(data_fit, epsilon = r) #predict_TPA(model_obj, data_fit, epsilon = r)

        ### Step 2.2: Construct Bootstrap Space Actuals on term t+1:t+n, with original lag terms and original model
        y_out_bs = model_obj.predict(new_data, epsilon = r) #predict_TPA(model_obj, new_data, epsilon = r)

        ### Stpe 2.3: Build Bootstrap-Space Model (M*)
        data_fit_bs = data_fit.copy()
        data_fit_bs.loc[num_lag_terms:, target_str] = y_actual_bs.tolist()
        
        # create an object of the same type
        model_bs = model_class()
        
        # if not retraining hyperparameters: use previously trained ones
        if not retrain_hyper_params:
            model_bs.set_hyper_params(model_obj.get_hyper_params())
            
        model_bs.train(data_fit_bs)   

        ### Step 2.4: Predict with M* on term t+1:t+n, with original lag terms
        bs_pred = model_bs.predict(new_data) #predict_TPA(model_bs, new_data)

        # Step 2.5: save the prediction error to error matrix
        error_matrix[k] = y_out_bs - bs_pred

        if (k+1) % 100 == 0:
            logger.info('{n} iterations done.'.format(n = k+1))
        
        if full_return:
            all_model_bs.append(model_bs)
    
    if not full_return:
        return y_pred, error_matrix
    else:
        return y_pred, error_matrix, all_model_bs, r




def get_quantile(value_matrix, quantile = 0.95):
    
    bs_iteration = len(value_matrix)
    pred_period = len(value_matrix[1])

    quantiles = []
    for i in range(pred_period):
        period_err = [value_matrix[k][i] for k in range(bs_iteration)]
        period_err = pd.Series(period_err)
        quantiles.append(period_err.quantile(q = quantile))
    
    quantiles = pd.Series(quantiles)

    return quantiles




def get_interval(value_matrix, upper_pct = 0.975, lower_pct = 0.025):
    
    bs_iteration = len(value_matrix)
    pred_period = len(value_matrix[1])

    lower = []
    upper = []
    for i in range(pred_period):
        period_err = [value_matrix[k][i] for k in range(bs_iteration)]
        period_err = pd.Series(period_err)
        lower.append(period_err.quantile(q = lower_pct))
        upper.append(period_err.quantile(q = upper_pct))
    
    lower = pd.Series(lower)
    upper = pd.Series(upper)
    return upper, lower






def plot_interval(train_date, train_y, new_date, new_y, new_pred, new_lower, new_upper, plot_title = '', save_path = None):
    
    train_date = train_date.reset_index(drop = True)
    train_y = train_y.reset_index(drop = True)
    new_date = new_date.reset_index(drop = True)
    new_y = new_y.reset_index(drop = True)
    new_pred = new_pred.reset_index(drop = True)
    new_lower = new_lower.reset_index(drop = True)
    new_upper = new_upper.reset_index(drop = True)
    
    train_dates = pd.DataFrame({'date': pd.to_datetime(train_date), 'train_y': train_y}) 
    new_dates = pd.DataFrame({'date': pd.to_datetime(new_date), 
                              'new_y': new_y,
                              'new_pred': new_pred,
                              'new_lower': new_lower,
                              'new_upper': new_upper})
    
    df = pd.concat([train_dates, new_dates], axis = 0, sort = False).reset_index(drop = True)
    df = df[['date', 'train_y', 'new_y', 'new_pred', 'new_lower', 'new_upper']]
    df = df.set_index('date')

    my_color=['lavender', 'lavender', 'grey', 'orangered', 'orangered']
    plt.figure()
    ax = df.plot(color = my_color, legend = False)
    plt.title(plot_title)
    ax.set_xlabel('')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    


    
def interval_performance(new_y, new_pred, new_lower, new_upper):
    
    # mean average percentage error
    mape = np.mean((new_y - new_pred)/new_y)

    # PI coverage probability
    coverage = float(np.sum((new_y > new_lower) & (new_y < new_upper))) / len(new_y)
    
    # percentage width is defined as the mean of interval width over the mean of actual value
    pwidth = np.mean((new_upper - new_lower))/np.mean(new_y)
    
    # PI normalized average width
    pinaw = np.mean(new_upper - new_lower)/(np.max(new_y) - np.min(new_y))
    
    return pd.DataFrame({'mape': [mape], 'coverage': [coverage], 'pwidth': [pwidth], 'pinaw': [pinaw]})




def derived_vars(fn, **kwargs):

    # find the minimum length
    lens = []
    for key, item in kwargs.items():
        lens.append(len(item))
    min_len = np.min(lens)

    # downsample to minimum length
    down_sampled_mx = {}
    for key, item in kwargs.items():
        if len(item) > min_len:
            down_sampled_mx[key] = random.sample(item, min_len)
        else:
            down_sampled_mx[key] = item

    # looping, get iteration item
    result_list = []
    for i in range(min_len):
        arg_dict = {}
        for key, item in down_sampled_mx.items():
            arg_dict[key] = item[i]

        result_list.append(fn(arg_dict))
    return(result_list)



def pred_interval_bs_shortcut(model_obj, all_model_bs, r, new_data):
    '''
    Args:
        model_obj: self-defined model trained in main space. With elements:
        all_model_bs: a list of model of the same class as model_obj, which are the bootstraped from model_obj
        r: fitted errors from model_obj
        
        new_data: data to predict, a pandas.DataFrame
        bs_iteration: number of bootstrap iterations

    Returns:
        y_pred: point prediction using main model
        error_matrix: a matrix[i][j] of the prediction error 
                      in each bootstrap iteration (dimension i), for each observation (dimension j).  
    '''
    
    bs_iteration = len(all_model_bs)
    pred_period = new_data.shape[0]
    # error matrix[i][j]: i is # bs iteration, j is # terms predicted with the model
    error_matrix = [[None for j in range(pred_period)] for i in range(bs_iteration)] 

    ### point prediction for t:t+n with original model (not used in bootstrap process)
    y_pred = model_obj.predict(new_data) #predict_TPA(model_obj, new_data)
    
    for k in range(bs_iteration):
        
        # retrieve bootstrap model M*
        model_bs = all_model_bs[k]
        ### Construct Bootstrap Space Actuals on term t+1:t+n, with original lag terms and original model
        y_out_bs = model_obj.predict(new_data, epsilon = r) #predict_TPA(model_obj, new_data, epsilon = r)

        ### Predict with M* on term t+1:t+n, with original lag terms
        bs_pred = model_bs.predict(new_data) #predict_TPA(model_bs, new_data)

        # Save the prediction error to error matrix
        error_matrix[k] = y_out_bs - bs_pred

        if (k+1) % 100 == 0:
            logger.info('{n} iterations done.'.format(n = k+1))
        
    return y_pred, error_matrix