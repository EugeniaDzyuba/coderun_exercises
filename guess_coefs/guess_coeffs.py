from typing import Callable

import math
import random

import numpy as np
import pandas as pd

from config import eps_a, eps_b, eps_c, init_coef_dict, lr

def calc_F(
    a: float,
    b: float,
    c: float,
    xi: float
)-> float:
    """
    Calculates function F for current a, b, c and xi

    Parameters:
    a (float): coefficient a,
    b (float): coefficient b,
    xi (float): x value
    
    Returns: value of F for current a, b, c and xi(float)
    """

    F = ((a + eps_a)*math.sin(xi) + (b + eps_b)*math.log(xi))**2 + (c+eps_c)*xi**2

    return F

def calc_K(
    a: float,
    b: float,
    xi: float
)-> float:
    """
    Calculates part of function F for current a, b and xi

    Parameters:
    a (float): coefficient a,
    b (float): coefficient b,
    xi (float): x value
    
    Returns: value of part of F for current a, b and xi (float)
    """

    K = (a + eps_a)*math.sin(xi) + (b + eps_b)*math.log(xi)

    return K

def DQi_da(
    a: float,
    b: float,
    c: float,
    xi: float,
    yi: float
)-> float:
    """
    Calculates gradient by a in one (xi,yi) point

    Parameters:
    a (float): coefficient a,
    b (float): coefficient b,
    c (float): coefficient c,
    xi (float): x value,
    yi (float): y value

    Returns: gradient by a in all (xi,yi) point (float)
    
    """
    dqi_da = (calc_F(a,b,c,xi)-yi)*2*calc_K(a,b,xi)*math.sin(xi)

    return dqi_da

def DQi_db(
    a: float,
    b: float,
    c: float,
    xi: float,
    yi: float
)-> float:
    """
    Calculates gradient by b in one (xi,yi) point

    Parameters:
    a (float): coefficient a,
    b (float): coefficient b,
    c (float): coefficient c,
    xi (float): x value,
    yi (float): y value

    Returns: gradient by b in all (xi,yi) point (float)
    
    """
    dqi_db = (calc_F(a,b,c,xi)-yi)*2*calc_K(a,b,xi)*math.log(xi)

    return dqi_db

def DQi_dc(
    a: float,
    b: float,
    c: float,
    xi: float,
    yi: float
)-> float:
    """
    Calculates gradient by c in one (xi,yi) point

    Parameters:
    a (float): coefficient a,
    b (float): coefficient b,
    c (float): coefficient c,
    xi (float): x value,
    yi (float): y value

    Returns: gradient by c in all (xi,yi) point (float)
    
    """
    dqi_dc = (calc_F(a,b,c,xi)-yi)*xi**2

    return dqi_dc

def calc_sum_grad(
    func: Callable,
    a: float,
    b: float,
    c: float,
    x: np.ndarray,
    y: np.ndarray
)-> float:
    
    """
    Calculates gradient by a, b or c in all (x,y) points

    Parameters:
    func (Callable): functions to calculate gradient in one point,
    a (float): coefficient a,
    b (float): coefficient b,
    c (float): coefficient c,
    x (np.ndarray): array of x values,
    y (np.ndarray): array of y values

    Returns: gradient by a, b or c in all (x,y) points (float)
    
    """
    
    sum_grad = 0
    for xi,yi in zip(x,y):
        sum_grad += func(a,b,c,xi,yi)

    return 2/len(x)*sum_grad

def calc_step(
    coef_dict: dict,
    learning_rate: float,
    x: np.ndarray,
    y: np.ndarray
)-> dict:
    """
    Calculates one step of gradient descent

    Parameters:
    X (np.ndarray): array of x values,
    y (np.ndarray): array of y values,
    coef_dict (dict): dict with coefficients a, b and c from previous iteration

    Returns: updated dict with coefficients a, b and c (dict)
    
    """

    a = coef_dict['a']
    b = coef_dict['b']
    c = coef_dict['c']

    coef_dict['a'] = a - learning_rate*calc_sum_grad(DQi_da,a,b,c,x,y)
    coef_dict['b'] = b - learning_rate*calc_sum_grad(DQi_db,a,b,c,x,y)
    coef_dict['c'] = c - learning_rate*calc_sum_grad(DQi_dc,a,b,c,x,y)
    
    return coef_dict

def calc_abs_residuals(
    X: np.ndarray,
    y: np.ndarray,
    coef_dict: dict
)-> float:
    """
    Calculates max residuals amoung all (x, y) pairs

    Parameters:
    X (np.ndarray): array of x values,
    y (np.ndarray): array of y values,
    coef_dict (dict): dict with coefficients a, b and c

    Returns: maximum residuals (float)
    """
    
    max_resid = 0
    for xi, yi in zip(X, y):
        if abs(calc_F(a = coef_dict['a'], b = coef_dict['b'], c = coef_dict['c'], xi = xi) - yi) >= max_resid:
            max_resid = calc_F(a = coef_dict['a'], b = coef_dict['b'], c = coef_dict['c'], xi = xi) - yi
            
    return max_resid

def run_grad_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    lr: float,
    init_coef_dict: dict,
    eps_a: float,
    eps_b: float,
    eps_c: float,
    max_abs_resid: float = 0.005,
    esr_const: int = 2000,
    max_iter: int = 10**6,
    lr_shrink_const: int = 10**3,
    lr_shrink_step: float = 0.95
)-> dict:
    
    """
    Runs gradient descent cycle

    Parameters:
    X_train (np.ndarray): array of x values used in optimization,
    y_train (np.ndarray): array of y values used in optimization,
    X_eval (np.ndarray): array of x values used in early stopping rounds mechanism,
    X_eval (np.ndarray): array of y values used in early stopping rounds mechanism,
    max_abs_resid (float): all residuals should be less then max_abs_resid for cycle to stop,
    esr_const (int): cycle will stop if residuals are not becoming less for this amount of iterations,
    max_iter (int): max amount of iterations in cycle,
    lr_shrink_const (int): learning rate is shrinked every lr_shrink_const iterations,
    lr_shrink_step (int): learning rate is shrinked by this value every lr_shrink_const iterations

    Returns:
    dict: dict with a,b,c values
    """
    
    new_step_coef_dict = init_coef_dict.copy()
    max_resid_min = 10**5
    
    counter = 0
    counter_esr = 0

    #while max_resid >= max_abs_resid:
    for counter in range(1, max_iter):
        # calc grad descent step
        new_step_coef_dict = calc_step(
            coef_dict = new_step_coef_dict,
            learning_rate = lr,
            x = X_train,
            y = y_train
        )

        
        # check residuals using eval dataset
        max_resid = calc_abs_residuals(X = X_eval, y = y_eval, coef_dict = new_step_coef_dict)

        # save min residuals for early stopping rounds mechanism
        if max_resid < max_resid_min:
            max_resid_min = max_resid
        else:
            counter_esr += 1 

        if counter_esr >= esr_const: #stop cycle if residuals are note becoming less for esr_const amount of iterations
            break

        if counter % lr_shrink_const == 0: #shrink learning rate once in lr_shrink_const iterations
            lr = lr*lr_shrink_step
            
    return new_step_coef_dict
    
if __name__ == '__main__':
    
    data = pd.read_csv('data.csv', header = None)
    data = data.sample(frac = 1, random_state = 5) #shuffle dataset
    
    X_train, y_train = data[:800][0].values, data[:800][1].values
    X_eval, y_eval = data[850:950][0].values, data[850:950][1].values
    X_test, y_test = data[950:][0].values, data[950:][1].values
    
    #initial values:
    
    eps_a = random.uniform(-0.001, 0.001)
    eps_b = random.uniform(-0.001, 0.001)
    eps_c = random.uniform(-0.001, 0.001)

    init_coef_dict = { #this values choosed by running code for some iterations with oter initial values
        'a':3.14,
        'b':2.72,
        'c':4
    }

    lr = 1*10**(-5)
    
    final_coef_dict = run_grad_descent(
        X_train = X_train,
        y_train = y_train,
        X_eval = X_eval,
        y_eval = y_eval,
        init_coef_dict = init_coef_dict,
        lr = lr,
        eps_a = eps_a,
        eps_b = eps_b,
        eps_c = eps_c
    )
    
    print('max test residuals:', calc_abs_residuals(X = X_test, y = y_test, coef_dict = final_coef_dict))
    print('coefs', final_coef_dict)

    
