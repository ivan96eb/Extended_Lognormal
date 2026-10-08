"""
Spherical-harmonic transforms with ducc0 (as in Galaxy_KaRMMa's karmma/transforms.py, without the autodiff wrappers).
alm use healpy's ordering (hp.Alm); maps are HEALPix RING.
"""

from functools import cache
from pathlib import Path
from urllib.request import urlretrieve

import ducc0
import healpy as hp
import numpy as np
from astropy.io import fits


@cache
def _geometry(nside):
    info = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    return {k: info[k] for k in ("theta", "phi0", "nphi", "ringstart")}


@cache
def pixel_weights(nside):
    """
    HEALPix full pixel weights (healpy's use_pixel_weights=True). The file comes from the healpy-data repository
    (the same source healpy uses) and is downloaded to ~/.cache/extended_lognormal/full_weights on first use.
    """
    name = f"healpix_full_weights_nside_{nside:04d}.fits"
    path = Path.home() / ".cache" / "extended_lognormal" / "full_weights" / name
    if not path.exists():
        print(f"Downloading pixel weights for nside = {nside} from healpy-data...")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urlretrieve(
                f"https://raw.githubusercontent.com/healpy/healpy-data/master/full_weights/{name}",
                path,
            )
        except Exception:
            path.unlink(missing_ok=True)
            raise
    with fits.open(path) as hdul:
        w8list = hdul[1].data.field(0).astype(np.float64)
    npix = hp.nside2npix(nside)
    w = np.zeros(npix)
    pnorth = vpix = 0
    for ring in range(2 * nside):
        qpix = min(ring + 1, nside)
        shifted = int(ring < nside - 1 or (ring + nside) % 2 == 1)
        qp4 = 4 * qpix
        for p in range(qp4):
            j4 = p % qpix
            w[pnorth + p] = w8list[vpix + min(j4, qpix - shifted - j4)]
        if ring < 2 * nside - 1:
            psouth = npix - pnorth - qp4
            w[psouth : psouth + qp4] = w[pnorth : pnorth + qp4]
        pnorth += qp4
        vpix += (qpix + 1) // 2 + 1 - ((qpix % 2) | shifted)
    return w + 1.0


def alm2map(alm, nside, lmax):
    """Synthesis: alm (nmaps, n_alm) -> maps (nmaps, npix). Exact for any lmax (evaluates the sum at the pixel centres)."""
    return ducc0.sht.synthesis(
        alm=alm[:, None, :],
        **_geometry(nside),
        lmax=lmax,
        mmax=lmax,
        spin=0,
        nthreads=0,
    )[:, 0, :]


def map2alm(maps, nside, lmax):
    """Analysis with full pixel weights, one pass (matches healpy.map2alm(..., use_pixel_weights=True))."""
    alm = ducc0.sht.adjoint_synthesis(
        map=(pixel_weights(nside) * maps)[:, None, :],
        **_geometry(nside),
        lmax=lmax,
        mmax=lmax,
        spin=0,
        nthreads=0,
    )
    return alm[:, 0, :] * (4 * np.pi / hp.nside2npix(nside))
