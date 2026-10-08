"""
Exact expected spectrum measured by `transforms.map2alm` on a HEALPix grid, for an isotropic field with power
out to lmax_in (above the measurement lmax): <C_hat_l> = sum_L K[l, L] C_L.

The field is sampled at the pixel centres and analysed in one pass with full pixel weights w~_p
(w~ = 4 pi / N_pix * weights), so a_hat_lm = sum_LM A_{lm,LM} a_LM with A_{lm,LM} = sum_p w~_p Y*_lm(p) Y_LM(p), and

    K[l, L] = 1/(2l+1) sum_m sum_M |A_{lm,LM}|^2.

K is computed exactly (no random fields), using the ring structure of the grid:
  * Y_lm = lambda_lm(theta) e^{i m phi} and the pixels of a ring are equally spaced in phi, so the phi sum is a
    Fourier coefficient of the weights around the ring: A_{lm,LM} = sum_r lambda_lm(theta_r) lambda_LM(theta_r) c_r(M - m),
    c_r(d) = sum_j w~_rj e^{i d phi_j}. The full weights repeat four times around each ring, so c_r(d) = 0 unless 4 | d.
  * Mirror rings (theta, pi - theta) have the same weights and phi_0, and lambda_lm(pi - theta) = (-1)^(l+m) lambda_lm(theta),
    so A vanishes unless l + m + L + M is even, and then only the northern rings and the equator are needed.
K includes both the in-band response (L <= lmax) and the aliasing of power from L > lmax.
"""
from pathlib import Path

import healpy as hp
import numpy as np

from .transforms import _geometry, pixel_weights

_RESCALE = 1e100   # the Legendre recursion is rescaled whenever |value| exceeds this, with a per-ring log scale


def _legendre_rings(m, lmax, cos_t, log_sin_t):
    """Normalized lambda_lm(theta_r) for l = m .. lmax, shape (lmax - m + 1, n_rings); sign convention irrelevant here."""
    out = np.zeros((lmax - m + 1, cos_t.size))
    k = np.arange(1, m + 1)
    log_s = 0.5 * (np.log((2 * m + 1) / (4 * np.pi)) + np.sum(np.log((2 * k - 1) / (2 * k)))) + m * log_sin_t
    p0 = np.ones_like(cos_t)                                  # lambda_mm / exp(log_s)
    out[0] = np.exp(log_s)
    if lmax == m:
        return out
    p1 = cos_t * np.sqrt(2 * m + 3) * p0                      # lambda_{m+1,m} / exp(log_s)
    out[1] = p1 * np.exp(log_s)
    l = np.arange(m + 2, lmax + 1)
    a = np.sqrt((4 * l**2 - 1) / (l**2 - m**2))
    b = np.sqrt(((l - 1)**2 - m**2) / (4 * (l - 1)**2 - 1))
    for i in range(l.size):
        p0, p1 = p1, a[i] * (cos_t * p1 - b[i] * p0)
        big = np.abs(p1) > _RESCALE
        if big.any():
            p0[big] /= _RESCALE
            p1[big] /= _RESCALE
            log_s[big] += np.log(_RESCALE)
        out[i + 2] = p1 * np.exp(log_s)
    return out


def _ring_setup(nside):
    """Northern rings plus the equator: cos theta, log sin theta, multiplicity (2 or 1), and the weights' ring DFTs."""
    geo = _geometry(nside)
    w = pixel_weights(nside) * (4 * np.pi / hp.nside2npix(nside))
    n_r = 2 * nside                                            # rings 0 .. 2 nside - 2 are northern, 2 nside - 1 is the equator
    theta, phi0 = geo["theta"][:n_r], geo["phi0"][:n_r]
    nphi, start = geo["nphi"][:n_r].astype(int), geo["ringstart"][:n_r].astype(int)
    w_hat = [np.fft.fft(w[s:s + n]).conj() for s, n in zip(start, nphi)]   # sum_j w_j e^{+2 pi i j k / n}
    mult = np.full(n_r, 2.0)
    mult[-1] = 1.0
    return np.cos(theta), np.log(np.sin(theta)), mult, phi0, nphi, w_hat


def _compute_kernel(nside, lmax, lmax_in, verbose):
    cos_t, log_sin_t, mult, phi0, nphi, w_hat = _ring_setup(nside)
    d_all = np.arange(-lmax_in - lmax, lmax_in + 1)
    # c_r(d) * multiplicity, for every shift d = M - m that can occur; shape (n_d, n_rings)
    c = np.array([mult[r] * np.exp(1j * d_all * phi0[r]) * w_hat[r][d_all % nphi[r]] for r in range(cos_t.size)]).T
    c_of = lambda d: c[d + lmax_in + lmax]
    lam_small = [_legendre_rings(m, lmax, cos_t, log_sin_t) for m in range(lmax + 1)]

    K = np.zeros((lmax + 1, lmax_in + 1))
    for M in range(lmax_in + 1):
        lam_M = _legendre_rings(M, lmax_in, cos_t, log_sin_t)            # rows L = M .. lmax_in
        for sgn in ((1,) if M == 0 else (1, -1)):
            for m in range(min(lmax, lmax_in) + 1):
                d = sgn * M - m
                if d % 4:
                    continue
                f_m = 1.0 if m == 0 else 2.0                              # (m, M) and (-m, -M) give equal |A|
                for pl in (0, 1):                                         # l + m parity; L + M must match it
                    lam_l = lam_small[m][pl::2]
                    lam_L = lam_M[pl::2]
                    if not lam_l.size or not lam_L.size:
                        continue
                    x = lam_l * c_of(d)                                   # (l, rings), complex
                    re, im = x.real @ lam_L.T, x.imag @ lam_L.T           # A for these (l, L)
                    K[m + pl::2, M + pl::2] += f_m * (re**2 + im**2)
        if verbose and (M % 256 == 0 or M == lmax_in):
            print(f"aliasing_kernel: M = {M} / {lmax_in}", flush=True)
    return K / (2 * np.arange(lmax + 1) + 1)[:, None]


def aliasing_kernel(nside=256, lmax=None, lmax_in=3 * 1024 - 1, cache=None, verbose=True):
    """
    Exact measurement kernel K (lmax + 1, lmax_in + 1) of `transforms.map2alm` at this nside and lmax (default
    3 nside - 1): an isotropic field with spectrum C_L (L <= lmax_in), sampled at the pixel centres, is measured as
    <C_hat_l> = K @ C. Includes the in-band response and the aliasing from L > lmax. Cached (compressed) in an .npz in the package's cache/ folder.
    """
    lmax = 3 * nside - 1 if lmax is None else lmax
    cache = Path(cache or Path(__file__).parent / "cache" / f"aliasing_kernel_nside{nside}_lmax{lmax}_in{lmax_in}.npz")
    if cache.exists():
        return np.load(cache)["K"]
    K = _compute_kernel(nside, lmax, lmax_in, verbose)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, K=K)
    return K
