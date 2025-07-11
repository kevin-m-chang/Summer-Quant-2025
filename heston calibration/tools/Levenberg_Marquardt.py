#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:07:29 2023

@author: ted
"""

"""
This module is a box-constrained Levenberg-Marquardt Algorithm for the Heston Model. 

"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from Heston_COS_METHOD import heston_cosine_method
from py_vollib_vectorized import vectorized_implied_volatility as calculate_iv
from scipy import linalg
from heston_derivative_constraints import heston_constraints,heston_implied_vol_derivative

def levenberg_Marquardt(Data,old_params,I,w,N,L,precision,params_2b_calibrated,accel_mag,min_acc, return_logs=False):
    """
  

    Parameters
    ----------
    Data : Class Object
        Contains all the information on the options. 
    old_params : NumPy Array
        Parameters to be calibrated.
    I : Int
        Number of iterations.
    w : Float
        Initial damping factor weight.
    N : Int
        Number of steps of summation in the COS-expansion.
    L : Float
        Range of truncation in the COS-expansion.
    v_bar : Float
        Long-Term vol.
    v0 : Float
        Initial vol.
    sigma : Float
        vol of vol.
    rho : Float
        Correlation between Stock and Volatility.
    kappa : Float
        Rate of mean-reversion.
    precision : Float
        precision of numerical differentiation
    params_2b_calibrated : list
        list of parameters (as str) you want to calibrate (as flags).
        E.g. if params_2b_calibrated = [v0,kappa,rho], then you are keeping v_bar and sigma constant.
    accel_mag : float
        the magnitude of the acceleration
    min_acc : float
        the minimum damping factor mu before the acceleration takes affect.

    Returns
    -------
    old_params : NumPy Array
        Calibrated parameters.
    counts_accepted : Int
        Number of iterations accepted.
    counts_rejected : Int
        number of iterations rejected.

    """
    
    # Foolproof check: skip if market_vol is all NaN or empty
    if (Data.market_vol is None or np.size(Data.market_vol) == 0 or np.all(np.isnan(Data.market_vol))):
        print("[LM] Skipping calibration: market_vol is all NaN or empty.")
        return old_params, 0, 0, [], []
    
    skip=1
    
    nu = 2
  
    eps_1 = 1e-5
    eps_2 = eps_1
    eps_3 = 0.0005

    new_price = heston_cosine_method(Data.S, Data.K, Data.T,N,L,Data.r,Data.q,old_params[0,0],old_params[1,0],old_params[2,0],old_params[3,0],old_params[4,0],Data.flag)


    new_vol = 100*calculate_iv(new_price,Data.S, Data.K, Data.T, Data.r, Data.flag, Data.q, model='black_scholes_merton',return_as='numpy') 

    f_x = (Data.market_vol - new_vol).reshape(np.size(new_vol),1)
    
    # Removing problematic options (give nan)
    f_x = Data.removing_nans_fx(f_x)
    
    J = -1*heston_implied_vol_derivative(Data.r,Data.K,Data.T,N,L,Data.q,Data.S,Data.flag,old_params[1,0],old_params[2,0],old_params[4,0],old_params[0,0],old_params[3,0], precision, params_2b_calibrated)

    # Removing problematic options (give nan)
    J, f_x = Data.removing_nans_J(J, f_x)
    
    M = np.size(Data.K)
    
    F_x = 0.5 * (1/M) * f_x.T @ f_x
    g = (1/M) * J @ f_x


    A = J@J.T 
    mu = w * np.amax(np.diag(J@J.T))
    print('mu: ', mu)


    counts_accepted=0
    counts_rejected=0
    k=0
    accelerator=1
    RMSE = []
    identity_matrix = np.eye(5)
    logs = []
    while k < I:
        
        # Calculating step of the parameters. inv is linalg.inv
        delta_params = linalg.inv(A + mu*identity_matrix) @ -g * (accelerator)
        
        new_params = heston_constraints(old_params + delta_params, old_params)
        
        # Calculating implied vol of new step
        new_price = heston_cosine_method(Data.S,Data.K,Data.T,N,L,Data.r,Data.q,new_params[0],new_params[4],new_params[1],new_params[2],new_params[3],Data.flag)
        new_vol = 100*calculate_iv(new_price,Data.S, Data.K, Data.T, Data.r, Data.flag, Data.q, model='black_scholes_merton',return_as='numpy')
        f_xh = (Data.market_vol - new_vol).reshape(np.size(new_vol),1)

        f_xh = Data.removing_nans_fx(f_xh)
        
        # Cost-Function of new step
        F_xh = 0.5 * (1/M) * f_xh.T@f_xh
        
        # Checking if step is better than the previous accepted step.
        gain_ratio = (F_x[0] - F_xh[0]) / (0.5*delta_params.T @ (mu*delta_params - g))
        
        if gain_ratio > 0:
            
            counts_accepted+=1
            
            old_params = new_params[:]
            
            J = -1*heston_implied_vol_derivative(Data.r,Data.K,Data.T,N,L,Data.q,Data.S,Data.flag,old_params[1,0],old_params[2,0],old_params[4,0],old_params[0,0],old_params[3,0], precision, params_2b_calibrated)
            
            J, f_xh = Data.removing_nans_J(J, f_xh)
            
            f_x = f_xh[:]
            F_x = F_xh[:]
            g = (1/M) * J @ f_x
            A = J@J.T 

            # Adjust damping factor
            mu = mu * np.maximum(1/3, 1 - (2 * gain_ratio - 1) ** 3)[0, 0]
            nu = 2
            
            RMSE.append(np.sqrt(2*F_x[0,0]))
            
            # Adjusting accelerator
            if mu < min_acc:
                accelerator = accel_mag*accelerator
            
            if k % 10 == 0:
                # Compose a log message for progress
                rmse_val = RMSE[-1] if RMSE else float('nan')
                msg = f"Iteration {k}: RMSE={rmse_val:.4f}, params={[float(x) for x in old_params.ravel()]}"

                logs.append(msg)
                if not return_logs:
                    print(msg)
                print(f"Iteration: {k}")
                print("Current params: vbar={:.6f}, sigma={:.6f}, rho={:.6f}, kappa={:.6f}, v0={:.6f}".format(
                    float(old_params[0,0]), float(old_params[1,0]), float(old_params[2,0]),
                    float(old_params[3,0]), float(old_params[4,0])
                ))
                print(f"  RMSE: {RMSE[-1]:.4f}")
                print(f"  Loss (F_x): {F_x[0,0]:.6f}")
                print(f"  mu: {mu:.6g}")
                if len(RMSE) > 10:
                    print(f"  Change in RMSE (last 10): {RMSE[-1] - RMSE[-10]:.4f}")
                print("-" * 60)


            if mu==np.inf:
                print('overflow')
                break
            
        else:

            counts_rejected +=1
            try:
                mu=mu*nu
                nu*=2
                accelerator=1
             
            # If we the damping factor goes off to infinity
            except:
                print("overflow")
                skip=0
                break
            if mu==np.inf:
                print("overflow")
                skip=0
                break
            continue

            
        if F_xh <= eps_1:
            print("Loss function close to zero")
            skip=0
            break
        
        if np.abs(np.amax(g))<=eps_2:
            print("Small J")
            skip=0
            break

        if k>=10:
            if np.abs(RMSE[k] - RMSE[k-5]) < eps_3:
                print("Steps converging to 0!")
                skip = 0
                break
        
        
        k+=1
        
    if skip==1:
        print('Exceeded maximum iterations')
    # Always return 5 outputs for compatibility
    return old_params, counts_accepted, counts_rejected, RMSE, logs
