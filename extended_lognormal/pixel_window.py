"""
Empirical pixel window of an N_side pixel-averaged map, out to high ell.

pixel_window() returns W(l), stored the same way as the old EmpWind pixwin_256.npy (the amplitude, not its square):
the pixel-averaged map has C_l^pix = W(l)^2 C_l, and W(l)^2 is the factor that multiplies (2l+1) C_l / 4pi in the pixel
variance. Unlike the old file, which is the window as read off the N_side map by anafast (it includes power from
l > 3 N_side aliased down, and depends on the spectrum), this is the pure pixel window. It is measured on a fine
grid, so the low-resolution map itself is never analysed (which is what limits an anafast-based measurement
to l ~ 3 N_side):

  1. draw Gaussian alm up to lmax from a smooth test spectrum (any spectrum works: the result is a ratio at each l);
  2. synthesise the field at N_side_fine;
  3. average it down to N_side (the mean of each pixel's children), then replicate each average back onto its children,
     which is hp.ud_grade twice;
  4. analyse that piecewise-constant map on the fine grid to get b_lm;
  5. the least-squares fit of b_lm = c(l) a_lm at each l is c(l) = sum_m Re(b_lm a_lm*) / sum_m |a_lm|^2 (the part of
     the pixel-averaging operator that is not diagonal in (l, m) is uncorrelated with a_lm and drops out).

The analysis in step 4 integrates over each pixel a second time, so c(l) is the *square* of the usual single pixel
window: c = W_healpy^2 for l <= 4 N_side - 1, where both exist (they agree to 0.3%). measure_pixel_window returns
that raw c = W^2; pixel_window smooths it and returns W = sqrt(c).
"""
from pathlib import Path

import ducc0
import healpy as hp
import numpy as np
from scipy.ndimage import uniform_filter1d

from .transforms import _geometry, alm2map


def _average_and_replicate(x, nside, nside_fine):
    """hp.ud_grade(hp.ud_grade(x, nside), nside_fine) for a RING map x, done in nested order (faster at N_side 4096)."""
    kids = (nside_fine // nside) ** 2
    average = hp.reorder(x, r2n=True).reshape(-1, kids).mean(axis=1)    # children of a pixel are contiguous in NESTED
    return hp.reorder(np.repeat(average, kids), n2r=True)


def measure_pixel_window(nside=256, lmax=3 * 1024 - 1, nside_fine=4096, n_realizations=20, seed=0, verbose=True):
    """
    Raw measurement of c(l) = W(l)^2, l = 0 .. lmax (set to 1 at l = 0, 1). About 20 s per realization at N_side_fine = 4096.
    The fine grid should resolve every mode: lmax <~ 0.75 nside_fine (the quadrature error of the pixel average
    grows as (l / nside_fine)^2).
    """
    rng = np.random.default_rng(seed)
    ell, m = hp.Alm.getlm(lmax)
    spectrum = np.zeros(lmax + 1)
    spectrum[2:] = 1.0 / (np.arange(2, lmax + 1) + 1.0) ** 2
    geometry, npix_fine = _geometry(nside_fine), hp.nside2npix(nside_fine)

    cross, auto = np.zeros(lmax + 1), np.zeros(lmax + 1)
    for r in range(n_realizations):
        alm = rng.standard_normal(ell.size) + 1j * rng.standard_normal(ell.size)
        alm[m > 0] *= np.sqrt(0.5)
        alm[m == 0] = alm[m == 0].real
        alm *= np.sqrt(spectrum[ell])

        x = alm2map(alm[None], nside_fine, lmax)[0]
        replicated = _average_and_replicate(x, nside, nside_fine)
        blm = ducc0.sht.adjoint_synthesis(map=replicated[None, None], **geometry, lmax=lmax, mmax=lmax, spin=0,
                                          nthreads=0)[0, 0] * (4 * np.pi / npix_fine)    # equal pixel areas
        cross += hp.alm2cl(blm, alm, lmax=lmax)
        auto += hp.alm2cl(alm, lmax=lmax)
        if verbose:
            print(f"pixel window: realization {r + 1}/{n_realizations}", flush=True)

    w2 = np.ones(lmax + 1)
    w2[2:] = cross[2:] / auto[2:]
    return w2


def pixel_window(nside=256, lmax=3 * 1024 - 1, smooth=41, cache=None, **kwargs):
    """
    W(l) for l = 0 .. lmax, such that C_pix = W^2 * C (same convention as healpy's pixwin and the old pixwin_256.npy).
    Cosmology independent, so the raw measurement of W^2 is cached on disk (first call takes about 20 s per
    realization). `smooth` is the width of a running mean applied to W^2 before the square root, to beat down its
    noise (the window is smooth; with 20 realizations the l-to-l scatter in W^2 is about 0.0011 above l = 1500,
    where W^2 ~ 0.01, before smoothing). Extra kwargs go to measure_pixel_window.
    """
    cache = Path(cache or Path(__file__).parent / f"pixel_window_nside{nside}_lmax{lmax}.npz")
    if cache.exists():
        raw = np.load(cache)["w2"]
    else:
        raw = measure_pixel_window(nside, lmax, **kwargs)
        np.savez(cache, w2=raw)
    w2 = uniform_filter1d(raw, size=smooth, mode="nearest") if smooth else raw
    return np.sqrt(np.maximum(w2, 0.0))
