import numpy as np
from scipy.optimize import minimize
from .Gn import Gn

def variance_from_Cl(Cl, ell_min=0):
    """
    Computes the variance of the field
    as predicted by the cls.
    
    Parameters
    ----------
    Cl : array
             Array of cls
    ell_min : float 
             First ell to start calculation. 
             Defaults to ell=0. 
    
    Returns
    -------
    variance : float
             Variance of the field as predicted by cls.
    """
    Cl = np.asarray(Cl)
    ell = np.arange(ell_min, ell_min + len(Cl))
    return np.sum((2*ell + 1) * Cl) / (4*np.pi)

def fit_gn(x_data, y_data, N, initial_params=None):
    """
    Fit a Gn transformation to (x, y) data points.
    
    Parameters
    ----------
    x_data : array
             array of x-values (standard normal)
    y_data : array 
             array of y-values (NL field)
    N : str
             Which G function to use ('2', '3', '4', '5')
    initial_params : array
             Initialization. If None, defaults to ones.
    
    Returns
    -------
    fitted_params : array
             Best fit parameters.
    """
    if initial_params is None:
        initial_params = np.ones((int(N)))
        #print(f"Dumb initialization for G{N}: {initial_params}")

    def cost_function(params):
        """Least squares cost"""
        try:
            y_pred = Gn(x_data, N, params)
            return np.sum((y_pred - y_data)**2)
        except:
            return np.inf
        
    result = minimize(
        fun=cost_function,
        x0=initial_params,
        method='BFGS'
    )
    
    return result.x

def fit_gn_with_constraint(x_data, y_data, N, cls, initial_params = None):
    """
    Fit a Gn transformation to (x, y) data points 
    in a self-consistent manner by also including
    the variance as predicted by the power spectrum.
    
    Parameters
    ----------
    x_data : array
             array of x-values (standard normal)
    y_data : array 
             array of y-values (NL field)
    N : str
             Which G function to use ('2' and '3' are the 
             only supported)
    cls : array
             The power spectrum of the field.
    initial_params : array
             Initialization. If None, defaults to ones.
    
    Returns
    -------
    fitted_params : array
             Best fit parameters.
    """
    var = variance_from_Cl(cls)
    def calc_constrained_params(unconstrained_params, var, N):
        if N == '2':
            beta = unconstrained_params[0]
            alpha = np.sqrt(np.log(1 + var / beta**2))
            return np.array([alpha, beta])
        elif N == '3':
            a, b = unconstrained_params
            c = np.sqrt((np.exp(a**2) - 1 + 2*a*b + b**2) / var) - 1
            return np.array([a, b, c])
        else:
            raise ValueError(f"Cannot do a constrained fit for G{N}. Use fit_gn_to_data instead.")
    def cost_function(unconstrained_params):
        """Least squares cost"""
        params = calc_constrained_params(unconstrained_params, var, N)
        try:
            y_pred = Gn(x_data, N, params)
            return np.sum((y_pred - y_data)**2)
        except:
            return np.inf
    if initial_params is None:
        if N == '2':
            beta_grid = np.geomspace(1e-6, 1, 50)
            cost_grid = np.array([cost_function([b]) for b in beta_grid])
            beta_init = beta_grid[np.argmin(cost_grid)]
            initial_params = np.array([beta_init, np.nan])
        elif N == '3':
            a_grid = np.linspace(0.01, 1.7, 25)
            b_grid = np.linspace(0.01, 2.7, 25)
            cost_grid = np.array([
                [cost_function([aa, bb]) for aa in a_grid]
                for bb in b_grid
            ])
            j_min, i_min = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
            initial_params = np.array([a_grid[i_min], b_grid[j_min], np.nan])
        else:
            initial_params = np.ones(int(N))
    initial_unconstrained_params = initial_params[:int(N)-1]
    #initial_unconstrained_params = np.ones_like(initial_unconstrained_params)
    result = minimize(
        fun=cost_function,
        x0=initial_unconstrained_params,
        method='BFGS'
    )
    return calc_constrained_params(result.x, var, N)