"""
Forward model for a planar double-spiral (bifilar) coplanar-electrode EIS sensor,
differential and single-channel measurement operators, and nuisance-marginalised
(efficient) Fisher information.

All quantities in SI units. Parameters of interest are in log coordinates so that
CRLBs are relative standard deviations.
"""
import numpy as np
from scipy.special import ellipk

EPS0 = 8.8541878128e-12

# ------------------------------------------------------------------ geometry
def kratio(w, s):
    """K(k)/K(k') for periodic coplanar strips (Olthuis et al. 1995)."""
    k = np.cos(np.pi * w / (2.0 * (w + s)))
    m = k * k
    return ellipk(m) / ellipk(1.0 - m)


def spiral_geometry(d):
    """d = (N, w, s, Dout). Returns dict of derived geometric quantities."""
    N, w, s, Dout = d
    p = 2.0 * (w + s)                       # radial pitch per turn (one trace of each electrode)
    Din = Dout - 2.0 * N * p                # inner clearance diameter
    Dbar = 0.5 * (Dout + Din)
    Le = np.pi * N * Dbar                   # trace length of one electrode
    Lg = (2.0 * N - 1.0) * np.pi * Dbar     # total facing (gap) length
    return dict(p=p, Din=Din, Dbar=Dbar, Le=Le, Lg=Lg, Ae=w * Le)


def cell_constant(d, kcorr=1.0):
    g = spiral_geometry(d)
    return kcorr * 2.0 * kratio(d[1], d[2]) / g["Lg"]


# ------------------------------------------------------------------ parameters
# parameters of interest (log coordinates)
THETA_NAMES = ["ln_sigma0", "ln_kappa", "ln_tau", "alpha"]
# kappa = (sigma_inf - sigma_0)/sigma_0 is the relative conductivity increment of the
# beta-dispersion, so that the dielectric increment is d_eps = kappa*sigma0*tau/eps0.
# nuisance parameters (prior mean in natural coordinates is given in NU0, prior sd in NU_SD)
NU_NAMES = ["ln_epsinf", "ln_Qs", "beta_s", "ln_Cfx", "ln_Rld", "ln_Lld", "dT",
            "ln_kc", "ln_Rsh", "dw_s", "ds_s", "dw_r", "ds_r", "ln_Qr", "beta_r"]

DEFAULT = dict(
    eps_sub=4.4,          # FR4/polyimide-class substrate relative permittivity
    alphaT=0.02,          # 1/K temperature coefficient of conductivity
    sigma_r=1.0,          # S/m reference buffer conductivity (known)
    epsr_r=78.0,          # reference permittivity (known)
)

NU0 = dict(ln_epsinf=np.log(70.0), ln_Qs=np.log(0.2), beta_s=0.88, ln_Cfx=np.log(5e-12),
           ln_Rld=np.log(2.0), ln_Lld=np.log(200e-9), dT=0.0, ln_kc=0.0, ln_Rsh=np.log(0.05),
           dw_s=0.0, ds_s=0.0, dw_r=0.0, ds_r=0.0, ln_Qr=np.log(0.2), beta_r=0.88)

NU_SD = dict(ln_epsinf=0.10, ln_Qs=0.50, beta_s=0.04, ln_Cfx=0.20, ln_Rld=0.30, ln_Lld=0.30,
             dT=0.5, ln_kc=0.02, ln_Rsh=0.15, dw_s=1.5e-6, ds_s=1.5e-6, dw_r=1.5e-6,
             ds_r=1.5e-6, ln_Qr=0.05, beta_r=0.01)

THETA_BOX = np.array([[np.log(0.1), np.log(1.6)],          # sigma0 [S/m]
                      [np.log(0.1), np.log(0.8)],          # relative conductivity increment
                      [np.log(8.0e-8), np.log(1.6e-6)],    # relaxation time [s] (fc 0.1-2 MHz)
                      [0.05, 0.25]])                       # Cole-Cole broadening
# target precision (1 s.d.) used to normalise the information matrix: 2% on sigma0,
# 10% on kappa and tau, 0.02 absolute on alpha
THETA_TOL = np.array([0.02, 0.10, 0.10, 0.02])
THETA_C = THETA_BOX.mean(axis=1)

# eta_rel: relative noise per Re/Im component of one acquisition channel (calibrated on the
# measured spectra of Section VI); eta_abs: absolute floor [ohm]; readout: 'digital' = two
# channels acquired independently and subtracted in software, 'analog' = differential
# front-end whose full scale is set by |dZ| with a CMRR-limited residual eta_cm*|Z|.
NOISE = dict(eta_rel=1.6e-3, eta_abs=0.05, eta_cm=2.0e-4, readout="digital")


def nu_vector(overrides=None):
    v = np.array([NU0[k] for k in NU_NAMES], float)
    if overrides:
        for k, val in overrides.items():
            v[NU_NAMES.index(k)] = val
    return v


def nu_sd_vector(scale=None):
    sd = np.array([NU_SD[k] for k in NU_NAMES], float)
    if scale:
        for k, f in scale.items():
            sd[NU_NAMES.index(k)] *= f
    return sd


# ------------------------------------------------------------------ physics
def eps_colecole(omega, deps, ln_tau, alpha, epsinf):
    a = alpha
    return epsinf + deps / (1.0 + (1j * omega * np.exp(ln_tau)) ** (1.0 - a))


def channel_impedance(omega, d, sigma_star, lnQ, beta, nu, chan, P, kcorr=1.0, extra=None):
    """Measured impedance of one channel (sample 's' or reference 'r')."""
    i = {k: n for n, k in enumerate(NU_NAMES)}
    N, w, s, Dout = d
    if chan == "s":
        w_ = w + nu[i["dw_s"]]; s_ = s + nu[i["ds_s"]]
    else:
        w_ = w + nu[i["dw_r"]]; s_ = s + nu[i["ds_r"]]
    # pitch is lithographically fixed: a width error is compensated by the gap
    dd = (N, w_, s_, Dout)
    g = spiral_geometry(dd)
    kc = np.exp(nu[i["ln_kc"]])
    Kc = kc * kcorr * 2.0 * kratio(w_, s_) / g["Lg"]
    Zmed = Kc / sigma_star
    Zep = 2.0 / (np.exp(lnQ) * g["Ae"] * (1j * omega) ** beta)
    Csub = EPS0 * P["eps_sub"] / Kc
    Zcell = 1.0 / (1.0 / (Zmed + Zep) + 1j * omega * Csub)
    Rtr = 2.0 * np.exp(nu[i["ln_Rsh"]]) * g["Le"] / (3.0 * w_)
    Zdev = Zcell + Rtr
    Cfx = np.exp(nu[i["ln_Cfx"]])
    Z = np.exp(nu[i["ln_Rld"]]) + 1j * omega * np.exp(nu[i["ln_Lld"]]) + 1.0 / (1.0 / Zdev + 1j * omega * Cfx)
    return Z


def sample_sigma_star(omega, th, nu, P, extra=None):
    i = {k: n for n, k in enumerate(NU_NAMES)}
    sig0 = np.exp(th[0]) * (1.0 + P["alphaT"] * nu[i["dT"]])
    deps = np.exp(th[1]) * np.exp(th[0]) * np.exp(th[2]) / EPS0
    eps = eps_colecole(omega, deps, th[2], th[3], np.exp(nu[i["ln_epsinf"]]))
    if extra is not None:   # unmodelled second (Debye) dispersion used only by the test generator
        eps = eps + extra[0] / (1.0 + 1j * omega * extra[1])
    return sig0 + 1j * omega * EPS0 * eps


def ref_sigma_star(omega, nu, P):
    i = {k: n for n, k in enumerate(NU_NAMES)}
    sig = P["sigma_r"] * (1.0 + P["alphaT"] * nu[i["dT"]])
    return sig + 1j * omega * EPS0 * P["epsr_r"]


def Z_sample(omega, d, th, nu, P=DEFAULT, kcorr=1.0, extra=None):
    i = {k: n for n, k in enumerate(NU_NAMES)}
    ss = sample_sigma_star(omega, th, nu, P, extra)
    return channel_impedance(omega, d, ss, nu[i["ln_Qs"]], nu[i["beta_s"]], nu, "s", P, kcorr)


def Z_ref(omega, d, nu, P=DEFAULT, kcorr=1.0):
    i = {k: n for n, k in enumerate(NU_NAMES)}
    sr = ref_sigma_star(omega, nu, P)
    return channel_impedance(omega, d, sr, nu[i["ln_Qr"]], nu[i["beta_r"]], nu, "r", P, kcorr)


def observe(omega, d, th, nu, op, P=DEFAULT, kcorr=1.0, extra=None):
    """Mean observable (complex) for operator op in {'single','diff'}."""
    Zs = Z_sample(omega, d, th, nu, P, kcorr, extra)
    if op == "single":
        return Zs
    return Zs - Z_ref(omega, d, nu, P, kcorr)


def noise_sd(omega, d, th, nu, op, P=DEFAULT):
    """Per-component (Re, Im) noise standard deviation at each frequency."""
    Zs = Z_sample(omega, d, th, nu, P)
    if op == "single":
        return NOISE["eta_rel"] * np.abs(Zs) + NOISE["eta_abs"]
    Zr = Z_ref(omega, d, nu, P)
    if NOISE["readout"] == "digital":
        v = (NOISE["eta_rel"] * np.abs(Zs) + NOISE["eta_abs"]) ** 2 \
            + (NOISE["eta_rel"] * np.abs(Zr) + NOISE["eta_abs"]) ** 2
        return np.sqrt(v)
    dZ = Zs - Zr
    return NOISE["eta_rel"] * np.abs(dZ) + NOISE["eta_cm"] * 0.5 * (np.abs(Zs) + np.abs(Zr)) + NOISE["eta_abs"]


# ------------------------------------------------------------------ Fisher information
def per_freq_info(omega, d, th, op, P=DEFAULT, nu=None, nu_mask=None, h=1e-5):
    """Per-frequency augmented FIM blocks G_l = J_l^T Sigma_l^{-1} J_l over (theta, nu_active).
    Returns array (M, q, q) and active nuisance indices."""
    if nu is None:
        nu = nu_vector()
    act = np.arange(len(NU_NAMES)) if nu_mask is None else np.asarray(nu_mask)
    x0 = np.concatenate([th, nu[act]])
    q = len(x0)
    sd = noise_sd(omega, d, th, nu, op, P)
    cols = []
    for k in range(q):
        step = h
        if k >= 4 and NU_NAMES[act[k - 4]].startswith(("dw", "ds")):
            step = 1e-9   # geometric tolerances are in metres
        xp = x0.copy(); xm = x0.copy(); xp[k] += step; xm[k] -= step
        thp, thm = xp[:4], xm[:4]
        nup = nu.copy(); num = nu.copy(); nup[act] = xp[4:]; num[act] = xm[4:]
        dz = (observe(omega, d, thp, nup, op, P) - observe(omega, d, thm, num, op, P)) / (2 * step)
        cols.append(dz)
    Jc = np.stack(cols, axis=1)                         # (M, q) complex
    Jr = np.stack([Jc.real, Jc.imag], axis=1) / sd[:, None, None]   # (M, 2, q)
    Jr[:, :, :4] *= THETA_TOL[None, None, :]       # tolerance-normalised parameters of interest
    G = np.einsum("mai,maj->mij", Jr, Jr)
    return G, act


def active_nuisance(op):
    """Nuisances that influence the observable of a given operator."""
    names = NU_NAMES[:9] + ["dw_s", "ds_s"]
    if op == "diff":
        names = names + ["dw_r", "ds_r", "ln_Qr", "beta_r"]
    return [NU_NAMES.index(n) for n in names]


def prior_precision(act, sd_scale=None):
    sd = nu_sd_vector(sd_scale)[act]
    return np.diag(1.0 / sd ** 2)


def efficient_info(A, Pnu, p=4):
    """Schur complement of the augmented information (with nuisance prior precision)."""
    Att = A[:p, :p]; Atn = A[:p, p:]; Ann = A[p:, p:] + Pnu
    return Att - Atn @ np.linalg.solve(Ann, Atn.T)


def naive_info(A, p=4):
    return A[:p, :p]
