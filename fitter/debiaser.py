import numpy as np 
import healpy as hp
from scipy.signal import savgol_filter
from multiprocessing import Pool
from .transform_cls import C_NG_to_C_G,diagnose_cl_G
from .mocker import get_y_maps,get_kappa,get_kappa_pixwin,get_kappa_pixwin_alms

def correct_mult_cl(cl,A):
    N_bins = cl.shape[0]
    cl_correct = np.zeros_like(cl)
    for i in range(N_bins):
        for j in range(i+1):
            cl_ij = np.sqrt(A[i]*A[j])*cl[i,j]
            cl_correct[i,j] = cl_ij
            cl_correct[j,i] = cl_ij
    return cl_correct

def cl_mock_avg(cl_NG,cl_G,fitted_params,pixwin,pixwin_ell_filter,N,auto=True,N_mocks=200,Nside=256,N_bins=4,gen_lmax=767):
    cl_arr  = np.zeros((N_mocks,N_bins,N_bins,gen_lmax+1))
    for mock in range(N_mocks):
        if mock % 50 == 0:
            print(f'Working on mock {mock}')
        y_maps, _      = get_y_maps(cl_G, Nside, N_bins, gen_lmax)
        kappa_mocklm     = get_kappa_pixwin_alms(y_maps, N_bins, N, fitted_params,Nside,pixwin_ell_filter)
        for i in range(N_bins):
            for j in range(i+1):
                if auto and i != j:
                    continue
                c_ij = hp.alm2cl(kappa_mocklm[i], kappa_mocklm[j], lmax=gen_lmax)
                cl_arr[mock,i,j] = c_ij
                cl_arr[mock,j,i] = c_ij
    perdiff_arr = np.zeros_like(cl_arr)
    for mock in range(N_mocks):
        perdiff_arr[mock]=(cl_arr[mock]/(cl_NG*pixwin**2)) 
    average_ratio = np.average(perdiff_arr,axis=0)
    return average_ratio

def Acoeff(average_ratio):
    Nbins = average_ratio.shape[0]
    beta = np.zeros(Nbins)
    for i in range(Nbins):
        beta[i]=np.average(average_ratio[i, i, 10:300])
    return 1/beta

def debiaser(cl_NG,N,params,pixwin,pixwinellfilter,N_iter=3,Nmocks=200):
    Nbins = cl_NG.shape[0]
    cl_NG_corr = cl_NG 
    for i in range(N_iter):
        print(f'Iteration {i}')
        cl_G       = C_NG_to_C_G(cl_NG_corr,params,Nbins,N)
        diagnose_cl_G(cl_G)
        avg_ratio = cl_mock_avg(cl_NG,cl_G,params,pixwin,pixwinellfilter,N,Nmocks)
        A         = Acoeff(avg_ratio)
        print('beta=',1/A)
        cl_NG_corr = correct_mult_cl(cl_NG_corr,A)
    return cl_NG_corr

def smooth_pixwin_savgol(pixwin, window_length=11, polyorder=3):
    """Savitzky-Golay smoothing - great for preserving peaks"""
    # Make sure window_length is odd and smaller than data length
    window_length = min(window_length, len(pixwin))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        return pixwin
    
    # Handle NaN values
    if np.any(~np.isfinite(pixwin)):
        mask = np.isfinite(pixwin)
        if np.sum(mask) < window_length:
            return pixwin
        
        # Interpolate over NaN values first
        pixwin_interp = np.interp(np.arange(len(pixwin)), 
                                np.arange(len(pixwin))[mask], 
                                pixwin[mask])
        return savgol_filter(pixwin_interp, window_length, polyorder)
    else:
        return savgol_filter(pixwin, window_length, polyorder)
    
def debiaser_premium(cl_NG,N,params,pixwin,pixwinellfilter,N_iter=3,Nmocks=200):
    Nbins = cl_NG.shape[0]
    cl_NG_corr = cl_NG 
    for i in range(N_iter):
        print(f'Iteration {i}')
        cl_G       = C_NG_to_C_G(cl_NG_corr,params,Nbins,N)
        diagnose_cl_G(cl_G)
        avg_ratio = cl_mock_avg(cl_NG,cl_G,params,pixwin,pixwinellfilter,N,Nmocks,N_bins=Nbins)
        smooth_bias = np.ones_like(avg_ratio)  
        # Only smooth diagonal terms since auto=True
        for i in range(Nbins):
            ratio_ii = smooth_pixwin_savgol(avg_ratio[i,i,2:2*256],window_length=50)
            smooth_bias[i,i,2:2*256] = ratio_ii
        print('beta=',1/Acoeff(avg_ratio))
        beta = np.array([smooth_bias[i,i] for i in range(Nbins)])
        A = 1/beta
        cl_NG_corr = correct_mult_cl(cl_NG_corr,A)
    return cl_NG_corr