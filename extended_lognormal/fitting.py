import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norm

from .gn_inv import _W, _X, gn_inv


def variance_from_cl(cl, lmin=2):
    """
    Variance of a field as predicted by its power spectrum,
    sigma^2 = sum_{l >= lmin} (2l + 1) C_l / (4 pi).

    Parameters
    ----------
    cl : array
            Power spectrum, starting at l = 0 along the last axis.
    lmin : int
            First multipole to include. Defaults to 2, since the
            maps have l = 0, 1 zeroed.

    Returns
    -------
    variance : float or array
            Variance (summed over the last axis of cl).
    """
    cl = np.asarray(cl)
    ell = np.arange(cl.shape[-1])
    return np.sum(((2 * ell + 1) * cl)[..., lmin:], axis=-1) / (4 * np.pi)


def gaussianize(field, n_bins=1000, x_range=(-4, 4)):
    """
    Pairs each pixel of a non-Gaussian field with its Gaussianized
    value x = Phi^{-1}(CDF), then averages within bins in x
    (the black points in FIG 1 of 2411.04759).

    Parameters
    ----------
    field : array
            Pixel values of the field.
    n_bins : int
            Number of equal-width bins in x.
    x_range : tuple
            (min, max) of x to keep.

    Returns
    -------
    x_mean : array
            Mean x of the pixels in each bin.
    y_mean : array
            Mean field value of the pixels in each bin.
    counts : array
            Number of pixels in each bin. Empty bins are dropped.
    """
    y = np.sort(np.ravel(field))
    x = norm.ppf((np.arange(y.size) + 0.5) / y.size)
    keep = (x >= x_range[0]) & (x <= x_range[1])
    x, y = x[keep], y[keep]

    edges = np.linspace(x[0], x[-1], n_bins + 1)
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    ok = counts > 0
    x_mean = np.bincount(idx, x, minlength=n_bins)[ok] / counts[ok]
    y_mean = np.bincount(idx, y, minlength=n_bins)[ok] / counts[ok]
    return x_mean, y_mean, counts[ok]


_WEIGHTS = {
    "sqrt": np.sqrt,
    "count": lambda n: n.astype(float),
    "uniform": lambda n: np.ones(len(n)),
}


# ---------------------------------------------------------------------------
# Each model maps its free parameters to full lbda with the variance
# constraint Var(gn_inv) = sigma^2 imposed. to_lbda returns a list of
# candidate lbda (empty if the constraint cannot be met).
# ---------------------------------------------------------------------------


class _G2:
    # free: alpha. beta from Var = beta^2 (e^{alpha^2} - 1)
    starts = ((0.2,), (0.5,), (1.0,))

    def __init__(self, s2):
        self.s2 = s2

    def to_lbda(self, free):
        alpha = free[0]
        v = np.expm1(alpha**2)
        if not v > 0:
            return []
        return [np.array([alpha, np.sqrt(self.s2 / v)])]

    def to_free(self, lbda):
        return np.array([lbda[0]])


class _G3:
    # free: a, b. c from Var = (e^{a^2} - 1 + 2ab + b^2) / (1 + c)^2
    starts = tuple((a, b) for a in (0.3, 0.8) for b in (0.1, 0.5))

    def __init__(self, s2):
        self.s2 = s2

    def to_lbda(self, free):
        a, b = free
        num = np.expm1(a**2) + 2 * a * b + b**2
        if not num > 0:
            return []
        return [np.array([a, b, np.sqrt(num / self.s2) - 1])]

    def to_free(self, lbda):
        return np.array(lbda[:2])


class _G4:
    # free: theta, log t, x0 with (a1, a2) = s (cos theta, sin theta).
    # log U = s Z(x) + const, so Var(s) = exp(K(2s) - 2K(s)) - 1 with
    # K(u) = log E[e^{uZ}] convex; Var is strictly increasing in s and
    # the root is unique. Solved by Newton in log s, warm-started from
    # the previous solution, with brentq as a fallback.
    starts = tuple(
        (np.arctan(r), np.log(t), x0)
        for r in (0.5, 1.0, 2.0)
        for t in (1.0, 3.0)
        for x0 in (0.0, 1.5)
    )

    def __init__(self, s2):
        self.s2 = s2
        self.log_s2 = np.log(s2)
        self.ls = 0.0

    @staticmethod
    def _cgf(u, z):
        # K(u) and K'(u) on the quadrature grid
        L = u * z
        m = L.max()
        e = _W * np.exp(L - m)
        E = e.sum()
        return np.log(E) + m, np.sum(e * z) / E

    def _g(self, ls, z):
        # log Var(s) - log sigma^2 and its derivative with respect to log s
        s = np.exp(ls)
        k1, dk1 = self._cgf(s, z)
        k2, dk2 = self._cgf(2 * s, z)
        d = k2 - 2 * k1
        var = np.expm1(d)
        return np.log(var) - self.log_s2, s * (var + 1) / var * 2 * (dk2 - dk1)

    def _solve(self, z):
        ls = self.ls
        for _ in range(20):
            g, dg = self._g(ls, z)
            if not (np.isfinite(g) and dg > 0):
                break
            step = np.clip(g / dg, -2.0, 2.0)
            ls -= step
            if abs(step) < 1e-10:
                self.ls = ls
                return np.exp(ls)
        # fallback: bracket the root and use brentq
        f = lambda l: self._g(l, z)[0]
        lo, hi = np.log(1e-6), np.log(10.0)
        while f(hi) < 0 and hi < np.log(1e3):
            hi += 1.0
        while f(lo) > 0 and lo > np.log(1e-10):
            lo -= 2.0
        if not f(lo) < 0 < f(hi):
            return None
        self.ls = brentq(f, lo, hi, xtol=1e-12)
        return np.exp(self.ls)

    def to_lbda(self, free):
        theta, log_t, x0 = free
        t = np.exp(log_t)
        c, s_ = np.cos(theta), np.sin(theta)
        z = c * _X + (s_ - c) / t * np.logaddexp(0.0, t * (_X - x0))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            s = self._solve(z)
        if s is None:
            return []
        return [np.array([s * c, s * s_, t, x0])]

    def to_free(self, lbda):
        a1, a2, t, x0 = lbda
        if not t > 0:
            raise ValueError(f"init for G4 needs t > 0, got t = {t}")
        return np.array([np.arctan2(a2, a1), np.log(t), x0])


class _G5:
    # free: a1, a2, log t, x0. With U = A(x) + b C(x), E[U] is linear and
    # E[U^2] quadratic in b, so E[U^2] = (1 + sigma^2) E[U]^2 is a quadratic
    # for b. Keep roots with E[U] > 0 (positive normalization).
    starts = tuple(
        (a1, a2, np.log(t), x0)
        for a1 in (0.3, 0.6)
        for a2 in (0.3, 0.8)
        for t in (1.0, 3.0)
        for x0 in (0.0, 1.5)
    )

    def __init__(self, s2):
        self.s2 = s2

    def to_lbda(self, free):
        a1, a2, log_t, x0 = free
        t = np.exp(log_t)
        with np.errstate(over="ignore", invalid="ignore"):
            brk = np.exp((a2 - a1) / t * np.logaddexp(0.0, t * (_X - x0)))
            A = np.exp(a1 * _X - 0.5 * a1**2) * brk
            C = _X * brk
            mA, mC = np.sum(_W * A), np.sum(_W * C)
            AA, AC, CC = np.sum(_W * A * A), np.sum(_W * A * C), np.sum(_W * C * C)
            q = 1 + self.s2
            q2, q1, q0 = CC - q * mC**2, 2 * (AC - q * mA * mC), AA - q * mA**2
            disc = q1**2 - 4 * q2 * q0
        if not np.all(np.isfinite([q2, q1, q0, disc])):
            return []
        if q2 == 0:
            roots = [-q0 / q1] if q1 != 0 else []
        else:
            if disc < 0:
                return []
            roots = [(-q1 + sg * np.sqrt(disc)) / (2 * q2) for sg in (1, -1)]
        return [np.array([a1, a2, b, t, x0]) for b in roots if mA + b * mC > 0]

    def to_free(self, lbda):
        a1, a2, _b, t, x0 = lbda
        if not t > 0:
            raise ValueError(f"init for G5 needs t > 0, got t = {t}")
        return np.array([a1, a2, np.log(t), x0])


_MODELS = {2: _G2, 3: _G3, 4: _G4, 5: _G5}


def _fit_one(x, y, w, N, model, starts, label):
    w = w / w.sum()

    def loss(lbda):
        with np.errstate(over="ignore", invalid="ignore"):
            val = np.sum(w * (gn_inv(x, N, lbda) - y) ** 2) / model.s2
        return val if np.isfinite(val) else np.inf

    def cost(free):
        return min((loss(l) for l in model.to_lbda(free)), default=np.inf)

    best = None
    for s0 in starts:
        # infeasible trial points have infinite cost; silence the inf - inf
        # warnings from the finite-difference gradient
        with np.errstate(invalid="ignore", over="ignore"):
            res = minimize(cost, np.asarray(s0, dtype=float), method="BFGS")
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError(
            f"G{N} fit failed for {label}: no start reached a finite cost"
        )
    return min(model.to_lbda(best.x), key=loss)


def fit_gn_inv(
    fields, N, sigma2, weight="sqrt", n_bins=1000, x_range=(-4, 4), init=None
):
    """
    Fits the G_N^{-1} transformation (see gn_inv) to each field, with the
    variance of the model constrained to sigma^2 so that it is consistent
    with the power spectrum.

    Parameters
    ----------
    fields : array
            Non-Gaussian maps, shape (Nbins, npix) or (npix,).
    N : int
            Which model to fit (2, 3, 4 or 5).
    sigma2 : float or array
            Target variance for each field, shape (Nbins,) or scalar.
            Typically variance_from_cl of the theory C_l.
    weight : str
            Weight of each x-bin in the least-squares loss:
                'sqrt'    : sqrt(pixel count) (default; compromise)
                'count'   : pixel count (best core of the PDF)
                'uniform' : equal (best far tails)
    n_bins : int
            Number of x-bins used by gaussianize.
    x_range : tuple
            Range of Gaussianized x kept in the fit.
    init : array, optional
            Starting lbda, shape (N,) or (N, Nbins), e.g. a previous fit.
            It and two jittered copies replace the default starting grid.

    Returns
    -------
    lbda : array
            Best-fit parameters, shape (N, Nbins), or (N,) for a single
            field. Rows follow gn_inv.
    """
    if N not in _MODELS:
        raise ValueError(f"Unknown model N={N}, expected one of {list(_MODELS)}")
    if weight not in _WEIGHTS:
        raise ValueError(f"Unknown weight '{weight}', expected one of {list(_WEIGHTS)}")

    fields = np.asarray(fields)
    single = fields.ndim == 1
    fields = np.atleast_2d(fields)
    n_fields = fields.shape[0]
    sigma2 = np.broadcast_to(np.asarray(sigma2, dtype=float), (n_fields,))
    if init is not None:
        init = np.asarray(init, dtype=float)
        init = np.broadcast_to(init.reshape(N, -1), (N, n_fields))
    rng = np.random.default_rng(0)

    lbda = np.empty((N, n_fields))
    for i in range(n_fields):
        x, y, counts = gaussianize(fields[i], n_bins, x_range)
        model = _MODELS[N](sigma2[i])
        if init is None:
            starts = model.starts
        else:
            f0 = model.to_free(init[:, i])
            starts = [f0] + [f0 + rng.normal(0, 0.15, f0.size) for _ in range(2)]
        lbda[:, i] = _fit_one(
            x, y, _WEIGHTS[weight](counts), N, model, starts, f"field {i}"
        )
    return lbda[:, 0] if single else lbda
