import numpy as np

# Fixed grid to compute E[f(X)] for X ~ N(0,1) with the trapezoid rule:
#     E[f(X)] ≈ np.sum(_W * f(_X))
# The grid never changes, so the normalization of G4/G5 is a smooth,
# deterministic function of the parameters (good for optimizers).
_X = np.linspace(-12.0, 12.0, 401)
_W = np.exp(-0.5 * _X**2) / np.sqrt(2.0 * np.pi) * (_X[1] - _X[0])


def _g2(x, alpha, beta):
    # beta * exp(alpha*x - alpha^2/2) - beta
    return beta * np.expm1(alpha * x - 0.5 * alpha**2)


def _g3(x, a, b, c):
    # n*(exp(a*x - a^2/2) + b*x + c) - 1 with n = 1/(1+c),
    # rewritten over the common denominator (the c's cancel)
    return (np.expm1(a * x - 0.5 * a**2) + b * x) / (1.0 + c)


def _g5_unnormalized(x, a1, a2, b, t, x0):
    # (exp(a1*x - a1^2/2) + b*x) * (1 + exp((x - x0)*t))^((a2 - a1)/t).
    # log1p(exp(.)) is ~10% faster than logaddexp but overflows for
    # t*(x - x0) > ~709; typical fits stay below ~25 even at the grid edge
    brk = np.exp((a2 - a1) / t * np.log1p(np.exp(t * (x - x0))))
    return (np.exp(a1 * x - 0.5 * a1**2) + b * x) * brk


def _g5_mean(a1, a2, b, t, x0):
    # E[U] over X ~ N(0,1); the G5 normalization is n = 1/E[U]
    return np.sum(_W * _g5_unnormalized(_X, a1, a2, b, t, x0), axis=-1, keepdims=True)


def _g5(x, a1, a2, b, t, x0):
    return _g5_unnormalized(x, a1, a2, b, t, x0) / _g5_mean(a1, a2, b, t, x0) - 1.0


def _g4_log_unnormalized(x, a1, a2, t, x0):
    # log of exp(a1*x - a1^2/2) * (1 + exp((x - x0)*t))^((a2 - a1)/t)
    # (log1p(exp(.)) rather than logaddexp for speed; see _g5_unnormalized)
    return a1 * x - 0.5 * a1**2 + (a2 - a1) / t * np.log1p(np.exp(t * (x - x0)))


def _g4(x, a1, a2, t, x0):
    # G4 is G5 with b = 0, but with no b*x term the whole thing is one
    # exponential: y = U/E[U] - 1 = expm1(log U - log E[U])
    log_mean = np.log(
        np.sum(
            _W * np.exp(_g4_log_unnormalized(_X, a1, a2, t, x0)), axis=-1, keepdims=True
        )
    )
    return np.expm1(_g4_log_unnormalized(x, a1, a2, t, x0) - log_mean)


_MODELS = {2: _g2, 3: _g3, 4: _g4, 5: _g5}


def gn_inv(x, N, lbda):
    """
    Maps a standard normal field x to a zero-mean non-Gaussian
    field y = G_N^{-1}(x), following Table 1 of arXiv:2411.04759.

        N = 2 : y = beta*exp(alpha*x - alpha^2/2) - beta
        N = 3 : y = n*(exp(a*x - a^2/2) + b*x + c) - 1,           n = 1/(1+c)
        N = 4 : y = n*exp(a1*x - a1^2/2)*B(x) - 1,                n = 1/E[...]
        N = 5 : y = n*(exp(a1*x - a1^2/2) + b*x)*B(x) - 1,        n = 1/E[...]

    with B(x) = (1 + exp((x - x0)*t))^((a2 - a1)/t). For N = 4, 5 the
    normalization n is computed numerically so that E[y] = 0.

    Parameters
    ----------
    x : array
            Standard normal values. Shape (Nbins, npix) when lbda has
            shape (N, Nbins); any shape when lbda has shape (N,).
    N : int
            Which model to use (2, 3, 4 or 5).
    lbda : array
            Model parameters, shape (N, Nbins) or (N,). Rows are, in order:
                N = 2 : alpha, beta
                N = 3 : a, b, c
                N = 4 : a1, a2, t, x0
                N = 5 : a1, a2, b, t, x0

    Returns
    -------
    y : array
            Transformed field, same shape as x.
    """
    if N not in _MODELS:
        raise ValueError(f"Unknown model N={N}, expected one of {list(_MODELS)}")
    lbda = np.asarray(lbda, dtype=float)
    if lbda.shape[0] != N:
        raise ValueError(
            f"lbda must have shape (N, ...) = ({N}, ...), got {lbda.shape}"
        )
    return _MODELS[N](np.asarray(x), *lbda[..., np.newaxis])
