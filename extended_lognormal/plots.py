"""Validation plots: one-point distributions and C_ell ratios of mocks against data / theory."""
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import colormaps, rcParams
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullFormatter

# The "default_jama" palette, written out so there's no pypalettes dependency.
JAMA = ["#374E55", "#DF8F44", "#00A1D5", "#B24745", "#79AF97", "#6A6599", "#80796B"]


def set_plot_style():
    """Apply the plot style (JAMA palette, LaTeX text, inward ticks on all sides) to matplotlib's rcParams."""
    cmap = ListedColormap(JAMA, name="default_jama")
    if cmap.name not in colormaps:
        colormaps.register(cmap)
    rcParams.update({
        "savefig.dpi": 200, "figure.dpi": 200, "font.size": 16,
        "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern"],
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.linewidth": 1.2, "lines.linewidth": 1.5,
        "legend.framealpha": 0.9, "legend.fontsize": 14,
        "image.cmap": cmap.name, "axes.prop_cycle": cycler(color=JAMA),
    })


def plot_histograms(data, mocks, N, titles=None, xlabel="field value", lin_range=(-1, 2), n_log=100, n_lin=200):
    """
    One-point distribution of the data against the mocks, one column per bin.

    data   (Nbins, Npix)           the data maps
    mocks  (n_mocks, Nbins, Npix)  the mock maps
    N      G_N model (only for the label)
    titles list of Nbins strings, one per column (optional)

    Top row: log scale over the full range. Bottom row: linear scale over lin_range. The dashed line is the mean
    histogram of the mocks and the band is their +-1 sigma scatter. Returns the figure.
    """
    n_mocks, n_bins = mocks.shape[:2]
    fig, axes = plt.subplots(2, n_bins, figsize=(8 * n_bins, 11), layout="constrained", squeeze=False)
    for i in range(n_bins):
        hi_log = max(data[i].max(), mocks[:, i].max())
        for row, (nb, scale, (lo, hi)) in enumerate(((n_log, "log", (lin_range[0], hi_log)), (n_lin, "linear", lin_range))):
            ax = axes[row, i]
            edges = np.linspace(lo, hi, nb + 1)
            # bins=int with range= takes numpy's fast uniform-bin path (about 4x faster than passing the edges array)
            c_data = np.histogram(data[i], bins=nb, range=(lo, hi))[0]
            c_mocks = np.array([np.histogram(mocks[k, i], bins=nb, range=(lo, hi))[0] for k in range(n_mocks)])
            mean, std = c_mocks.mean(0), c_mocks.std(0)

            # the data line is the thick one underneath; the thinner mock line is drawn on top of it
            ax.stairs(mean + std, edges, baseline=np.maximum(mean - std, 1e-1), fill=True, color=JAMA[1], alpha=0.35, zorder=1)
            ax.stairs(c_data, edges, color=JAMA[0], lw=2.5, label="Data (pixel average)", zorder=2)
            ax.stairs(mean, edges, color=JAMA[1], ls="--", label=rf"$G_{N}$ mocks (mean of {n_mocks})", zorder=3)

            if scale == "log":
                ax.set_yscale("log")
                ax.set_ylim(1e-1, 2 * max(c_data.max(), mean.max()))
            else:
                ax.set_ylim(0, 1.2 * max(c_data.max(), mean.max()))
            ax.set_xlim(lo, hi)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Pixels per bin")
            if row == 0 and titles is not None:
                ax.set_title(titles[i])
            ax.legend(loc="upper right")
    return fig


def plot_cl_ratios(cl_mock, cl_mock_err, cl_theory, N, n_mocks, lmin=2, ell_mark=512):
    """
    Mean mock spectrum over theory, lower triangle of an (Nbins, Nbins) grid of panels.

    cl_mock, cl_mock_err, cl_theory   (Nbins, Nbins, lmax + 1): mock mean, its standard error, theory as measured
    N, n_mocks                        only for the label
    ell_mark                          dotted vertical line (the highest l used downstream)

    The band is the standard error of the mock mean. Auto-spectrum panels span 1 +- 6%; the cross-spectrum panels
    get their own limits since they are noisier. Returns the figure.
    """
    n_bins, lmax = cl_mock.shape[0], cl_mock.shape[-1] - 1
    ell = np.arange(lmax + 1)[lmin:]
    fig, axes = plt.subplots(n_bins, n_bins, figsize=(9 * n_bins, 5.5 * n_bins), sharex=True, layout="constrained", squeeze=False)
    for i in range(n_bins):
        for j in range(n_bins):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            r = cl_mock[i, j, lmin:] / cl_theory[i, j, lmin:]
            err = cl_mock_err[i, j, lmin:] / cl_theory[i, j, lmin:]
            ax.fill_between(ell, r - err, r + err, color=JAMA[1], alpha=0.35)
            ax.plot(ell, r, color=JAMA[1], label=rf"$G_{N}$ mocks (mean of {n_mocks})")
            ax.axhline(1, color="k", ls="--", lw=1)
            ax.axvline(ell_mark, color="k", ls=":", lw=1.2)
            half = 0.06 if i == j else max(0.06, 1.1 * np.max(np.abs(r - 1)))
            ax.set_xscale("log")
            ax.set_xlim(lmin, lmax + 1)
            ax.set_ylim(1 - half, 1 + half)
            ticks = [lmin, 10, 100, ell_mark, lmax]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(rf"$C_\ell^{{{j}{i}}}$ / theory (as measured)")
            if i == n_bins - 1:
                ax.set_xlabel(r"$\ell$")
    axes[0, 0].legend(loc="lower left")
    return fig
