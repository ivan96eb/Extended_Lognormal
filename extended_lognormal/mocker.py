"""Mocks: correlated Gaussian maps y from C_G, then field_i = G_i(y_i) pixel by pixel."""

import os
from concurrent.futures import ThreadPoolExecutor

import healpy as hp
import numpy as np
from scipy.stats import skew

from .gn_inv import _W, _X, gn_inv
from .transforms import alm2map


class Mocker:
    """
    Parameters
    ----------
    cl_g : array (Nbins, Nbins, lmax + 1)
            Spectra of the Gaussian fields y (transform_cls.gaussianize_cl). lmax sets the number of modes.
    lam : array (N, Nbins)
            G_N parameters from fit_gn_inv.
    N : int
            G model (2, 3, 4 or 5).
    nside : int
            Nside of the output maps.
    verbose : bool
            Print diagnostics: a setup summary and the expected variance of y at construction, then
            the measured moments of y and of the field at each call (compared to what the model implies).
    """

    def __init__(self, cl_g, lam, N, nside, verbose=False):
        self.nbins, self.lmax = cl_g.shape[0], cl_g.shape[-1] - 1
        self.lam, self.N, self.nside, self.verbose = lam, N, nside, verbose
        cl = np.moveaxis(cl_g, 2, 0)  # (l, Nbins, Nbins)
        active = np.any(
            cl != 0, axis=(1, 2)
        )  # all-zero multipoles (l < 2, truncated tail) carry no power
        min_eig = np.linalg.eigvalsh(cl[active]).min(axis=1)
        if np.any(min_eig <= 0):
            bad = np.flatnonzero(active)[min_eig <= 0]
            raise ValueError(
                f"C_G is not positive definite at {bad.size} multipoles, first l = {bad[:10].tolist()}"
            )
        chol = np.zeros_like(cl)
        chol[active] = np.linalg.cholesky(cl[active])
        ell, self._emm = hp.Alm.getlm(self.lmax)
        self._chol = np.ascontiguousarray(
            chol[ell].transpose(1, 2, 0)
        )  # factor at each alm index, (Nbins, Nbins, n_alm)
        self._n_threads = min(32, os.cpu_count() or 1)

        # what the truncated C_G implies for y, and for the field if y ~ N(0, var_y)
        e = (2 * np.arange(self.lmax + 1) + 1) / (4 * np.pi)
        cov_y = np.einsum("l,ijl->ij", e, cl_g)  # l = 0, 1 are zero in cl_g
        self.var_y = np.diag(cov_y).copy()
        self.corr_y = cov_y / np.sqrt(np.outer(self.var_y, self.var_y))
        if verbose:
            print(
                f"Mocker: {self.nbins} bins, nside {nside}, lmax {self.lmax} "
                f"({'above' if self.lmax > 3 * nside - 1 else 'within'} 3*nside-1 = {3 * nside - 1}), G{N}"
            )
            print(
                f"  expected var(y) per bin: {np.round(self.var_y, 4)}   (1 = nothing lost to the lmax cut)"
            )
            print(
                f"  expected y correlation between bins: {np.round(self.corr_y[np.tril_indices(self.nbins, -1)], 4)}"
            )

    def draw_xlm(self, rng=None, n_mocks=None):
        """
        Latent xlm in healpy alm ordering and Galaxy_KaRMMa's convention: real parts iid N(0, 1) (all l),
        imaginary parts iid N(0, 1) for m > 0 and zero for m = 0. The 1/sqrt(2) that makes them unit-power
        alm is applied in apply_cl. (KaRMMa's xlm_real / xlm_imag are the l > 1 and l > 1, m > 0 entries.)

        Returns shape (Nbins, n_alm), or (n_mocks, Nbins, n_alm) for an integer n_mocks. For n_mocks each
        mock gets its own child generator of rng (rng.spawn), so they are drawn in parallel; the result
        is still deterministic given rng's state.
        """
        rng = np.random.default_rng() if rng is None else rng

        def draw(g):
            imag = g.standard_normal((self.nbins, self._emm.size))
            imag[:, self._emm == 0] = 0.0
            return g.standard_normal((self.nbins, self._emm.size)) + 1j * imag

        if n_mocks is None:
            return draw(rng)
        with ThreadPoolExecutor(self._n_threads) as ex:
            return np.stack(list(ex.map(draw, rng.spawn(n_mocks))))

    def apply_cl(self, xlm):
        """
        Latent xlm (..., Nbins, n_alm) -> alm of y with spectra C_G: y_lm,i = sum_j L_l,ij x_lm,j,
        with 1/sqrt(2) for m > 0 (so unit power) and real for m = 0, as in Galaxy_KaRMMa.
        """
        ylm = sum(self._chol[:, j] * xlm[..., j : j + 1, :] for j in range(self.nbins))
        ylm[..., self._emm > 0] *= np.sqrt(0.5)
        ylm[..., self._emm == 0] = ylm[..., self._emm == 0].real
        return ylm

    def get_y_maps(self, xlm=None, rng=None, n_mocks=None):
        """
        Gaussian maps y, shape (Nbins, npix), or (n_mocks, Nbins, npix). Pass xlm (from draw_xlm, with or
        without a leading mock axis) to reproduce mocks; otherwise they are drawn with rng
        (a fresh default_rng() if neither is given).
        """
        if xlm is None:
            xlm = self.draw_xlm(rng, n_mocks)
        alm = self.apply_cl(xlm)
        y = alm2map(alm.reshape(-1, alm.shape[-1]), self.nside, self.lmax).reshape(
            alm.shape[:-1] + (-1,)
        )
        if self.verbose:
            y3 = y.reshape(-1, self.nbins, y.shape[-1])
            cov = np.einsum("mip,mjp->ij", y3, y3) / (y3.shape[0] * y3.shape[2])
            corr = (cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov))))[
                np.tril_indices(self.nbins, -1)
            ]
            print(
                f"y maps{f' ({y3.shape[0]} mocks pooled)' if y.ndim == 3 else ''}: mean {np.round(y3.mean((0, 2)), 4)}   "
                f"var {np.round(y3.var(2).mean(0), 4)} (expected {np.round(self.var_y, 4)})   corr {np.round(corr, 4)}"
            )
        return y

    def apply_G(self, y):
        """G_i(y_i) in each bin, pixel by pixel. y has shape (..., Nbins, npix); maps are done in parallel."""
        flat = y.reshape(-1, y.shape[-1])
        f = lambda q: gn_inv(flat[q], self.N, self.lam[:, q % self.nbins])
        if len(flat) == 1:
            out = f(0)[None]
        else:
            with ThreadPoolExecutor(self._n_threads) as ex:
                out = np.array(list(ex.map(f, range(len(flat)))))
        out = out.reshape(y.shape)
        if self.verbose:
            out3 = out.reshape(-1, self.nbins, y.shape[-1])
            for i in range(self.nbins):
                d = out3[:, i].ravel()
                g = gn_inv(
                    np.sqrt(self.var_y[i]) * _X, self.N, self.lam[:, i]
                )  # y ~ N(0, var_y), on gn_inv's grid
                mean = _W @ g
                var = _W @ (g - mean) ** 2
                sk = _W @ (g - mean) ** 3 / var**1.5
                print(
                    f"field bin {i}: mean {d.mean():+.4f}  var {d.var():.5f}  skew {skew(d):.3f}  "
                    f"min {d.min():+.3f}  max {d.max():.2f}\n"
                    f"       implied by the model for y ~ N(0, var_y): mean {mean:+.4f}  var {var:.5f}  skew {sk:.3f}"
                )
        return out

    def mocks(self, n_mocks, rng=None, chunk_size=10):
        """
        Generator over n_mocks mocks in chunks of chunk_size, each (chunk, Nbins, npix). Memory stays at one
        chunk, so use this instead of __call__ when only a statistic of each mock is needed. Diagnostics
        (verbose) are printed for the first chunk only. About 10 mocks per chunk is already as fast as it gets.
        """
        rng = np.random.default_rng() if rng is None else rng
        verbose, self.verbose = self.verbose, False
        try:
            for start in range(0, n_mocks, chunk_size):
                self.verbose = verbose and start == 0
                yield self.apply_G(
                    self.get_y_maps(rng=rng, n_mocks=min(chunk_size, n_mocks - start))
                )
        finally:
            self.verbose = verbose

    def __call__(self, xlm=None, rng=None, n_mocks=None):
        """Mock field, shape (Nbins, npix), or (n_mocks, Nbins, npix) (all held in memory: see mocks())."""
        if xlm is not None or n_mocks is None:
            return self.apply_G(self.get_y_maps(xlm, rng))
        return np.concatenate(list(self.mocks(n_mocks, rng)))
