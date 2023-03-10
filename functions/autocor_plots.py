import matplotlib.pyplot as plt
import numpy as np
import torch
from statsmodels.graphics.tsaplots import plot_acf
import arviz

# ---  Create the necessary function for the autocorrelation plot
def autocor_plots(X_chain, method_str, nLags=100):
    '''
    Inputs:
        - X_chain (Matrix): Markov chain
        - var (matrix): Variance of the Markov chain computed in the Fourier domain.
        - M=method_str (string): Name of the Markov chain used
        - nLags (int): Number of lags used for the autocorrelation function.
    Outputs:
        - Autocorrelation plots. we also highligthed the effective sample size for each trace.
    '''
    
    # --- Checking whether the samples size is greater than the number of lags considered.
    if X_chain.shape[0] < nLags:
        raise ValueError(f"nLags must be smaller than the number of samples!")
    
   # --- 
    if torch.is_tensor(X_chain):
        X_chain = X_chain.detach().cpu().numpy()
    
    # --- Vectorise the the Markov chain
    X_chain_vec = X_chain.reshape(len(X_chain),-1)
    
    # --- Variance of the Markov chain
    var_sp = np.var(X_chain_vec, axis = 0)
    
    # --- lower trace of the Markov chain
    trace_elem_max_variance = X_chain_vec[:,np.argmax(var_sp)]
    
     # --- Faster trace of the Markov chain
    trace_elem_min_variance = X_chain_vec[:,np.argmin(var_sp)]
    
     # --- Medium-speed trace of the Markov chain
    ind_medium = np.argsort(var_sp)[len(var_sp)//2]
    trace_elem_median_variance = X_chain_vec[:,ind_medium]
    
    # --- effective sample size
    e_slow = ESS(trace_elem_max_variance.reshape(-1))
    e_fast = ESS(trace_elem_min_variance.reshape(-1))
    e_med  = ESS(trace_elem_median_variance.reshape(-1))

    # --- Here we are generating the autocorrelation function for these three traces: lower, medium and faster.
    _,ax = plt.subplots(figsize=(15,10))
    plot_acf(trace_elem_median_variance,ax=ax,label=f'Median-speed, ESS = {e_med: .2f}',alpha=None,lags=nLags)
    plot_acf(trace_elem_max_variance,ax=ax,label=f'Slowest, ESS = {e_slow: .2f}',alpha=None,lags=nLags)
    plot_acf(trace_elem_min_variance,ax=ax,label=f'Fastest, ESS = {e_fast: .2f}',alpha=None,lags=nLags)
    handles, labels= ax.get_legend_handles_labels()
    handles=handles[1::2]
    labels =labels[1::2]
    ax.set_title(method_str)
    ax.set_ylabel("ACF")
    ax.set_xlabel("lags")
    ax.set_ylim([-1.1,1.3])
    ax.legend(handles=handles, labels=labels,loc='best',shadow=True, numpoints=1)
    
    
def ESS(arr):
    '''
    Input: 
        - arr (vector): This vector contains the trace of the Markov chain
    Output:
        - ess (float): effective sample size of the trace.
    '''
    ess = arviz.ess(arr)
    return ess