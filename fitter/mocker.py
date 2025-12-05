import numpy as np 
import healpy as hp 
from .Gn import Gn

def eigvec_matmul(A, x, nbins):
    y = np.zeros_like(x)
    for i in range(nbins):
        for j in range(nbins):
            y[i] += A[i,j] * x[j]
    return y

def apply_cl(xlm, cl, gen_lmax,nbins):

    ell, emm = hp.Alm.getlm(gen_lmax)
    L = np.linalg.cholesky(cl.T).T
    
    xlm_real = xlm.real
    xlm_imag = xlm.imag
    
    L_arr = np.swapaxes(L[:,:,ell[ell > -1]], 0,1)
    
    ylm_real = eigvec_matmul(L_arr, xlm_real,nbins) / np.sqrt(2.)
    ylm_imag = eigvec_matmul(L_arr, xlm_imag,nbins) / np.sqrt(2.)

    ylm_real[:,ell[emm==0]] *= np.sqrt(2)
    
    return ylm_real + 1j * ylm_imag

def get_xlm(xlm_real, xlm_imag,gen_lmax,nbins):
    ell, emm = hp.Alm.getlm(gen_lmax)
    #==============================
    _xlm_real = np.zeros((nbins, len(ell)))
    _xlm_imag = np.zeros_like(_xlm_real)
    _xlm_real[:,ell > 1] = xlm_real
    _xlm_imag[:,(ell > 1) & (emm > 0)] = xlm_imag
    xlm = _xlm_real + 1j * _xlm_imag
    #==============================
    return xlm
    
def generate_xlm(nbins,gen_lmax):
    ell, emm = hp.Alm.getlm(gen_lmax)
    xlm_real = np.random.normal(size=(nbins, (ell > 1).sum()))
    xlm_imag = np.random.normal(size=(nbins, ((ell > 1) & (emm > 0)).sum()))

    xlm = get_xlm(xlm_real, xlm_imag,gen_lmax,nbins)
    return xlm, [xlm_real,xlm_imag]

def generate_mock_y_lm(cl,nbins,gen_lmax,xlms=None):
    if xlms is not None:
        #print('xlms spec')
        xlm = xlms
        _xlm = None
    else:
        #print('xlms not spec')
        xlm,_xlm = generate_xlm(nbins,gen_lmax)
    return apply_cl(xlm, cl,gen_lmax,nbins), _xlm

def get_y_maps(cl,nside,nbins,gen_lmax,xlms=None):
    y_lm,xlm = generate_mock_y_lm(cl,nbins,gen_lmax,xlms)
    y_maps = []
    for i in range(nbins):
        y_map = hp.alm2map(np.ascontiguousarray(y_lm[i]), nside, lmax=gen_lmax, pol=False)
        y_maps.append(y_map)    
    return np.array(y_maps),xlm    

def get_kappa(y_maps,nbins,N,fitted_params):
    k_list = []
    for i in range(nbins):
        k_nf = Gn(y_maps[i], N, fitted_params[i])
        k = k_nf
        k_list.append(k)  
    k_arr  = np.array(k_list)
    return k_arr  

def get_kappa_pixwin(y_maps,nbins,N,fitted_params,nside,pixwinatell):
    k_list = []
    lmax = 2*nside
    for i in range(nbins):
        k_nf = Gn(y_maps[i], N, fitted_params[i])
        k = k_nf
        klm = hp.map2alm(k,lmax=lmax)
        klm = klm * pixwinatell 
        k = hp.alm2map(klm,nside)
        k_list.append(k)  
    k_arr  = np.array(k_list)

    return k_arr  

def get_kappa_pixwin_alms(y_maps,nbins,N,fitted_params,nside,pixwinatell):
    klm_list = []
    lmax = 2*nside
    for i in range(nbins):
        k_nf = Gn(y_maps[i], N, fitted_params[i])
        k = k_nf
        klm = hp.map2alm(k,lmax=lmax)
        klm = klm * pixwinatell 
        klm_list.append(klm)  
    k_arr  = np.array(klm_list)

    return k_arr  