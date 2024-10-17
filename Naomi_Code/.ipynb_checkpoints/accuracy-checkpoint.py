# 7/30/23 | Script to compute the accuracy of the CNN results for CGM properties.

import numpy as np
import math
from permetrics.regression import RegressionMetric
from sklearn.metrics import r2_score
from scipy.stats import chisquare

def rmse(y_actual, y_predicted):

    rmse_value = np.sqrt(np.mean(np.square(y_actual - y_predicted)))
    return rmse_value

def rel_mean_err(y_actual, y_predicted):
    
    evaluator = RegressionMetric(y_actual, y_predicted, decimal=5)
    return(evaluator.MRE(multi_output="raw_values"))

def R_squared(y_actual, y_predicted):

    R2 = r2_score(y_actual, y_predicted)
    return R2

def chi_squared(y_actual, y_predicted, error, epsilon=1e-10):
    if len(y_predicted) != len(y_actual):
        raise ValueError("Both observed and expected must have the same length.")

    chi_sq = (y_predicted - y_actual)**2 / error**2

    return chi_sq

