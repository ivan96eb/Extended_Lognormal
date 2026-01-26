import numpy as np 
from scipy.stats import norm


def get_binned_data(x_gaussianized, y_data, x_range,n_bins):
    """
    Bin x values and compute mean y within each bin
    
    Parameters
    ----------
    x_gaussianized : array of Gaussianized x values
    y_data : array of corresponding y values
    n_bins : number of bins
    x_range : tuple (min, max) to keep 
    
    Returns
    -------
    x_bin_centers : x value at center of each bin
    y_bin_means : mean y value in each bin
    """
    # Keep only values between -5 and 5
    mask = (x_gaussianized >= x_range[0]) & (x_gaussianized <= x_range[1])
    x_filtered = x_gaussianized[mask]
    y_filtered = y_data[mask]
    
    # Create bins
    bin_edges = np.linspace(x_filtered.min(), x_filtered.max(), n_bins + 1)
    
    # Find which bin each x value belongs to
    bin_indices = np.digitize(x_filtered, bin_edges) - 1
    
    # Clip to valid range
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute mean y for each bin
    x_bin_centers = []
    y_bin_means = []
    
    for i in range(n_bins):
        bin_mask = (bin_indices == i)
        if np.sum(bin_mask) > 0:  # Only include bins with data
            x_bin_centers.append(bin_edges[i:i+2].mean())  # center of bin
            y_bin_means.append(y_filtered[bin_mask].mean())
    
    return np.array(x_bin_centers), np.array(y_bin_means)

def empirical_cdf(map):
    """
    Computes the CDF of a map
   
    Parameters
    ----------
    map : array of that represents map
   
    Returns
    -------
    sorted_map : x-coordinates of the CDF
    cdf_values : y-coordinates of the CDF
    """
    sorted_map = np.sort(map)
    cdf_values = np.arange(1, len(sorted_map) + 1) / len(sorted_map)
    cdf_values = np.clip(cdf_values, 1e-10, 1 - 1e-10)
   
    return sorted_map, cdf_values

def histogramer2d(map,Nbins,x_range=(-4.5,4.5)):
    """
    Given a NL field, computes the black triangles in
    FIG 1 in 2411.04759
    
    Parameters
    ----------
    map : field that we want to model
    
    Returns
    -------
    x_avg : x value of triangle (standard normal)
    y_avg : y value of triangle (NL field)
    """
    y_data, cdf  = empirical_cdf(map)
    x_gaussianized = norm.ppf(cdf)
    x_avg,y_avg = get_binned_data(x_gaussianized,y_data,x_range,Nbins)
    return x_gaussianized,x_avg,y_avg