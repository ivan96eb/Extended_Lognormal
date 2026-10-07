"""
From the non-Gaussian C_ell of delta to the C_ell of the Gaussian field y that G_N maps onto it
(upgrade of fitter/transform_cls.py: exact Hermite-Mehler series instead of a lookup table).

    C_NG -> xi_NG(theta) -> xi_G(theta) = F^{-1}(xi_NG(theta)) -> C_G
"""
import numpy as np
from scipy.special import legendre_p_all, roots_legendre

from .gn_inv import _W, _X, gn_inv

_NMAX = 60            # Hermite terms computed per bin
_PARSEVAL_TOL = 1e-8  # series is cut where the missing variance sum_{k>n} c_k^2 is below this fraction


def _hermite(u, n):
    """Normalized probabilists' Hermite polynomials He_k(u)/sqrt(k!), k = 0..n."""
    H = np.empty((n + 1,) + np.shape(u))
    H[0], H[1] = 1.0, u
    for k in range(1, n):
        H[k + 1] = (u * H[k] - np.sqrt(k) * H[k - 1]) / np.sqrt(k + 1)
    return H


_HW = _hermite(_X, _NMAX) * _W   # projection onto Hermite polynomials on gn_inv's fixed grid


class Mehler:
    """
    Correlation transfer F_ij(rho) between y_i, y_j (unit variance, correlation rho) and delta_i = G_i(y_i),
    delta_j = G_j(y_j) (Mehler): F(rho) = sum_{n>=1} c_n^i c_n^j rho^n, c_n = Hermite coefficients of G.

    G2 and G3 are analytic. With k = a_i a_j:
        G2: F(rho) = b_i b_j (exp(k rho) - 1)                                          (and F^-1 is a logarithm)
        G3: F(rho) = [exp(k rho) - 1 + m rho] / [(1 + c_i)(1 + c_j)],  m = (a_i + b_i)(a_j + b_j) - k
    (G3 = [(exp(a x - a^2/2) - 1) + b x] / (1 + c): its Hermite coefficients are a^n / sqrt(n!) for n >= 2 and a + b for
    n = 1, all over 1 + c, and the n >= 2 terms sum to an exponential.) G3's inverse is found by Newton.
    Otherwise the series is cut where Parseval says both bins' missing variance is below _PARSEVAL_TOL (cap _NMAX).
    """

    def __init__(self, N, lam_i, lam_j):
        self.N = N
        if N == 2:
            self.k = lam_i[0] * lam_j[0]       # alpha_i alpha_j
            self.s = lam_i[1] * lam_j[1]       # beta_i beta_j
        elif N == 3:
            self.k = lam_i[0] * lam_j[0]                                            # a_i a_j
            self.m = (lam_i[0] + lam_i[1]) * (lam_j[0] + lam_j[1]) - self.k         # (a_i + b_i)(a_j + b_j) - a_i a_j
            self.d = (1.0 + lam_i[2]) * (1.0 + lam_j[2])
            self._slope0 = (self.k + self.m) / self.d                               # F'(0)
        else:
            c = [_HW @ gn_inv(_X, N, lam) for lam in (lam_i, lam_j)]
            n = max(self._n_terms(ci) for ci in c)
            self.pc = c[0][1:n + 1] * c[1][1:n + 1]   # coefficient of rho^k is pc[k-1]
            self._slope0 = self.pc[0]
        self.lo, self.hi = self.F(np.array([-1.0, 1.0]))

    @staticmethod
    def _n_terms(c):
        power = c[1:] ** 2
        missing = power.sum() - np.cumsum(power)      # variance beyond term n (n = 1, 2, ...)
        ok = missing < _PARSEVAL_TOL * power.sum()
        return int(np.argmax(ok)) + 1 if ok.any() else len(power)

    def F(self, r):
        return self.dF(r)[0]

    def dF(self, r):
        """F(r) and F'(r)."""
        if self.N == 2:
            e = self.s * np.exp(self.k * r)
            return e - self.s, self.k * e
        if self.N == 3:
            return (np.expm1(self.k * r) + self.m * r) / self.d, (self.k * np.exp(self.k * r) + self.m) / self.d
        f, d = np.zeros_like(r), np.zeros_like(r)     # Horner for f = sum pc[k-1] r^(k-1), F = r f
        for p in self.pc[::-1]:
            d = d * r + f
            f = f * r + p
        return f * r, f + d * r

    def inv(self, t):
        """y correlation that maps to delta correlation t."""
        t = np.clip(t, self.lo, self.hi)
        if self.N == 2:
            return np.clip(np.log1p(t / self.s) / self.k, -1, 1)
        r = np.clip(t / self._slope0, -1, 1)
        for _ in range(80):
            f, df = self.dF(r)
            r = np.clip(r - (f - t) / df, -1, 1)
        return r


def gaussianize_cl(cl_ng, lam, N, lmax=None, nodes_per_ell=2, on_npd="zero"):
    """
    Power spectra of the Gaussian fields y whose G_N transforms have spectra cl_ng.

    Parameters
    ----------
    cl_ng : array (Nbins, Nbins, l_in + 1)
            Non-Gaussian auto and cross spectra, starting at l = 0.
    lam : array (N, Nbins)
            G_N parameters from fit_gn_inv (each G_i assumes var(y_i) = 1).
    N : int
            G model (2, 3, 4 or 5).
    lmax : int, optional
            Highest l of the output (default l_in). May exceed l_in: xi_G is built from cl_ng and
            then projected onto more multipoles, which is how to generate y with more modes.
    nodes_per_ell : int, optional
            Gauss-Legendre nodes per multipole (nodes_per_ell * max(l_in, lmax) in total). 2 is converged to
            ~1e-6 of the peak C_G for pixel-windowed spectra, but only ~1e-2 for unwindowed ones.
    on_npd : {"error", "truncate", "zero"}, optional
            What to do if C_G(l) is not positive definite (NPD) at some l >= 2, so the Cholesky factor the mocks
            need doesn't exist there. The shape of the output never changes.
              "error"    : raise a ValueError naming the multipoles.
              "truncate" : set C_G to exactly zero from the first NPD multipole up (default).
              "zero"     : set C_G to exactly zero at the NPD multipoles only.
            For the last two, what was done and how much variance of y it removed is printed.

    Returns
    -------
    cl_g : array (Nbins, Nbins, lmax + 1)
            Gaussian spectra, with l = 0, 1 set to zero (the maps have no monopole or dipole), plus any multipoles
            zeroed by on_npd. Mocker treats all-zero multipoles as no power.
    """
    nbins, l_in = cl_ng.shape[0], cl_ng.shape[-1] - 1
    lmax = l_in if lmax is None else lmax
    mu, w = roots_legendre(nodes_per_ell * max(l_in, lmax))
    P_in, P_out = (legendre_p_all(l, mu).squeeze() for l in (l_in, lmax))
    e_in = (2 * np.arange(l_in + 1) + 1) / (4 * np.pi)

    cl_g = np.zeros((nbins, nbins, lmax + 1))
    for i in range(nbins):
        for j in range(i + 1):
            xi_ng = (e_in * cl_ng[i, j]) @ P_in
            xi_g = Mehler(N, lam[:, i], lam[:, j]).inv(xi_ng)
            cl_g[i, j] = cl_g[j, i] = 2 * np.pi * (P_out @ (w * xi_g))
    cl_g[:, :, :2] = 0

    if on_npd not in ("error", "truncate", "zero"):
        raise ValueError(f'on_npd must be "error", "truncate" or "zero", got {on_npd!r}')
    bad = np.flatnonzero(np.linalg.eigvalsh(np.moveaxis(cl_g, 2, 0)).min(axis=1) <= 0)
    bad = bad[bad >= 2]                                  # l = 0, 1 are zero on purpose
    if bad.size:
        listed = f"l = {bad[:10].tolist()}" + (" ..." if bad.size > 10 else "")
        if on_npd == "error":
            raise ValueError(f"C_G is not positive definite at {bad.size} multipoles, {listed}")
        zeroed = np.arange(bad[0], lmax + 1) if on_npd == "truncate" else bad
        e = (2 * np.arange(lmax + 1) + 1) / (4 * np.pi)
        lost = np.einsum("l,il->i", e[zeroed], np.diagonal(cl_g, axis1=0, axis2=1).T[:, zeroed])
        removed = ", ".join(f"{x:+.4f} (bin {i})" for i, x in enumerate(lost))
        if on_npd == "truncate":
            print(f"gaussianize_cl: C_G is not positive definite from l = {bad[0]} ({bad.size} multipoles fail); "
                  f"setting C_G = 0 for l >= {bad[0]} (requested lmax {lmax}, effective lmax {bad[0] - 1}).")
        else:
            print(f"gaussianize_cl: C_G is not positive definite at {bad.size} multipoles, {listed}; "
                  f"setting C_G = 0 at those multipoles only.")
        print(f"  variance of y removed: {removed}")
        cl_g[:, :, zeroed] = 0
    return cl_g


def check_positive_definite(cl_g):
    """Smallest eigenvalue of the (Nbins, Nbins) matrix at each l >= 2; prints the l where it is <= 0."""
    min_eig = np.linalg.eigvalsh(np.moveaxis(cl_g, 2, 0)[2:]).min(axis=1)
    bad = np.flatnonzero(min_eig <= 0) + 2
    if bad.size:
        print(f"Not positive definite at {bad.size} multipoles, first: {bad[:10].tolist()}")
    else:
        print(f"Positive definite at every l >= 2 (smallest eigenvalue {min_eig.min():.3e})")
    return min_eig
