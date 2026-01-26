import numpy as np 
from .gauss_hermite import get_gh_nodes_weights 

def Gn(x, n, params,N_nodes = 20):
    """
    Evaluate Gn transformation at given x values
    
    Parameters
    ----------
    x : array of x values (standard normal inputs)
    n : str, which G function to use ('2', '3', '4', '5')
    params : array of parameters for the transformation
    N_nodes : number of Gauss-Hermite quad points to compute
             integral for normalization.
    
    Returns
    -------
    y : transformed values
    """

    def compute_normalization(params):
        """Compute n such that E[n*arg - 1] = 0"""
        if n == '2':
            return None
        elif n == '3':
            return None
        elif n == '4':
            a1, a2, t, x0 = params
            quad_points, quad_weights = get_gh_nodes_weights(N_nodes)
            arg1 = np.exp(a1*quad_points - 0.5*a1**2)
            arg2 = (1 + np.exp((quad_points - x0)*t))**((a2-a1)/t)
            arg = quad_weights * arg1 * arg2
            return 1/np.sum(arg)
        elif n == '5':
            a1, a2, b, t, x0 = params
            quad_points, quad_weights = get_gh_nodes_weights(N_nodes)
            arg1 = np.exp(a1*quad_points - 0.5*a1**2) + b*quad_points
            arg2 = (1 + np.exp((quad_points - x0)*t))**((a2-a1)/t)
            arg = quad_weights * arg1 * arg2
            return 1/np.sum(arg)
    
    # Evaluate transformation
    if n == '2':
        alpha, beta = params
        return beta * np.exp(alpha * x - 0.5 * alpha**2) - beta
    
    elif n == '3':
        a, b, c = params
        arg = np.exp(a * x - 0.5 * a**2) + b*x + c
        norm = 1/(1+c)
        return norm * arg - 1
    
    elif n == '4':
        a1, a2, t, x0 = params
        arg1 = np.exp(a1*x - 0.5*a1**2)
        arg2 = (1 + np.exp((x - x0)*t))**((a2-a1)/t)
        arg = arg1 * arg2
        norm = compute_normalization(params)
        return norm * arg - 1
    
    elif n == '5':
        a1, a2, b, t, x0 = params
        arg1 = np.exp(a1*x - 0.5*a1**2) + b*x
        arg2 = (1 + np.exp((x - x0)*t))**((a2-a1)/t)
        arg = arg1 * arg2
        norm = compute_normalization(params)
        return norm * arg - 1
    
    else:
        raise ValueError(f"Unknown model type: {n}")