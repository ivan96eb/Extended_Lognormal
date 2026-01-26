import numpy as np
from numpy.polynomial.hermite import hermgauss

def get_gh_nodes_weights(n_nodes):
    """
    Computes the quadrature points and weights to 
    solve integrals for the form I = E[f(x)] where x
    is standard normal. So, to solve integral
    simply do np.sum(y_weights * f(y_nodes)).
    
    Parameters
    ----------
    x : int
            Number of quadrature points
    
    Returns
    -------
    points_weights : tuple
            Returns the quadrature points, weights.
    """
    t, w = hermgauss(n_nodes)
    y_nodes = np.sqrt(2.0) * t
    y_weights = w / np.sqrt(np.pi)
    return y_nodes, y_weights