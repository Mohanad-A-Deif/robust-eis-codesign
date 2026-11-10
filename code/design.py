"""Robust geometry--frequency co-design: scenario handling, convex frequency design,
geometry search, baselines and worst-case evaluation."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import itertools
import model as M

FGRID = np.logspace(3, 7, 64)            # 1 kHz .. 10 MHz candidate grid
OMEGA = 2 * np.pi * FGRID
D0 = (5, 50e-6, 50e-6, 3.0e-3)           # nominal (template) geometry
GEOM_LB = np.array([20e-6, 20e-6, 1.5e-3])
GEOM_UB = np.array([200e-6, 200e-6, 5.0e-3])
N_RANGE = range(2, 13)
DIN_MIN = 0.5e-3


def feasible(d):
    g = M.spiral_geometry(d)
    return g["Din"] >= DIN_MIN - 1e-12


# ------------------------------------------------------------------ information blocks
class InfoCache:
    def __init__(self, op, sd_scale=None, naive=False):
        self.op = op
        self.act = M.active_nuisance(op)
        self.P = M.prior_precision(self.act, sd_scale)
        self.naive = naive
        self.cache = {}

    def G(self, d, th):
        key = (tuple(np.round(d, 12)), tuple(np.round(th, 10)), M.NOISE["readout"])
        if key not in self.cache:
            G, _ = M.per_freq_info(OMEGA, d, th, self.op, nu_mask=self.act)
            if len(self.cache) > 20000:
                self.cache.clear()
            self.cache[key] = G
        return self.cache[key]

    def eff(self, G, wts):
        A = np.tensordot(wts, G, axes=(0, 0))
        if self.naive:
            return A[:4, :4]
        return M.efficient_info(A, self.P)


def lam_min(I):
    return np.linalg.eigvalsh(I)[0]


def crit(I, kind="E"):
    if kind == "E":
        return lam_min(I)
    sgn, ld = np.linalg.slogdet(I + 1e-9 * np.eye(4))
    return ld


# ------------------------------------------------------------------ frequency design (convex)
def proj_capped_simplex(y, m):
    lo, hi = y.min() - 1.0, y.max()
    for _ in range(80):
        tau = 0.5 * (lo + hi)
        s = np.clip(y - tau, 0, 1).sum()
        if s > m:
            lo = tau
        else:
            hi = tau
    return np.clip(y - 0.5 * (lo + hi), 0, 1)


def _value_grad(Gs, P, wts, kind, mu_rel=30.0, naive=False):
    """Soft-min (over scenarios and eigenvalues) of the efficient information and its
    (super)gradient with respect to the frequency weights. Gs: (T, M, q, q)."""
    Gs = np.asarray(Gs)
    A = np.einsum("l,tlij->tij", wts, Gs)
    Att, Atn, Ann = A[:, :4, :4], A[:, :4, 4:], A[:, 4:, 4:] + P[None]
    if naive:
        X = np.zeros((A.shape[0], Ann.shape[1], 4)); S = Att
    else:
        X = np.linalg.solve(Ann, np.transpose(Atn, (0, 2, 1)))
        S = Att - Atn @ X
    lams, V = np.linalg.eigh(S)
    U = np.concatenate([V, -X @ V], axis=1)            # (T, q, 4)
    GU = np.matmul(Gs, U[:, None, :, :])                # (T, M, q, 4)
    Q = np.einsum("tqi,tlqi->tli", U, GU)                # d lambda_{t,i} / d w_l
    if kind == "A":   # average-case (Bayesian) E-criterion over the scenario set
        T = lams.shape[0]
        lm = lams.min(1); mu = mu_rel / np.maximum(lm, 1e-12)
        e = np.exp(-mu[:, None] * (lams - lm[:, None])); Z = e.sum(1)
        vals = lm - np.log(Z) / mu
        pi = e / Z[:, None]
        grad = np.einsum("tli,ti->l", Q, pi) / T
        return vals.mean(), grad, lm.mean()
    if kind == "E":
        lmin = lams.min()
        mu = mu_rel / max(lmin, 1e-12)
        e = np.exp(-mu * (lams - lmin)); Z = e.sum(); pi = e / Z
        val = lmin - np.log(Z) / mu
        grad = np.einsum("tli,ti->l", Q, pi)
        return val, grad, lmin
    ld = np.log(np.maximum(lams, 1e-300)).sum(1)
    lm = ld.min(); mu = 5.0
    e = np.exp(-mu * (ld - lm)); Z = e.sum(); pi = e / Z
    val = lm - np.log(Z) / mu
    grad = np.einsum("tli,ti->l", Q / lams[:, None, :], pi[:, None].repeat(4, 1))
    return val, grad, lm


def design_weights(Gs, P, m, kind="E", iters=200, naive=False, w0=None, rtol=1e-6):
    Mg = Gs[0].shape[0]
    w = np.full(Mg, m / Mg) if w0 is None else w0.copy()
    val, g, _ = _value_grad(Gs, P, w, kind, naive=naive)
    step = 0.5 * m / (np.abs(g).max() + 1e-30)
    best = (val, w.copy())
    for it in range(iters):
        while True:
            wn = proj_capped_simplex(w + step * g, m)
            vn, gn, _ = _value_grad(Gs, P, wn, kind, naive=naive)
            if vn >= val - 1e-12 or step < 1e-14:
                break
            step *= 0.5
        if np.abs(wn - w).max() < 1e-7 or (vn - val) < rtol * abs(val) and it > 20:
            w, val, g = wn, vn, gn
            if val > best[0]:
                best = (val, w.copy())
            break
        w, val, g = wn, vn, gn
        step *= 1.3
        if val > best[0]:
            best = (val, w.copy())
    return best[1]


def subset_value(Gs, P, idx, kind="E", naive=False):
    Gs = np.asarray(Gs)
    A = Gs[:, list(idx)].sum(1)
    if naive:
        S = A[:, :4, :4]
    else:
        Att, Atn, Ann = A[:, :4, :4], A[:, :4, 4:], A[:, 4:, 4:] + P[None]
        S = Att - Atn @ np.linalg.solve(Ann, np.transpose(Atn, (0, 2, 1)))
    lam = np.linalg.eigvalsh(S)
    if kind == "E":
        return lam[:, 0].min()
    if kind == "A":
        return lam[:, 0].mean()
    return np.log(np.maximum(lam, 1e-300)).sum(1).min()


def round_topm(w, m):
    return tuple(sorted(np.argsort(-w)[:m]))


def swap_refine(Gs, P, idx, kind="E", naive=False, max_pass=3):
    idx = list(idx); best = subset_value(Gs, P, idx, kind, naive)
    Mg = Gs[0].shape[0]
    for _ in range(max_pass):
        improved = False
        for pos in range(len(idx)):
            for cand in range(Mg):
                if cand in idx:
                    continue
                trial = idx.copy(); trial[pos] = cand
                v = subset_value(Gs, P, trial, kind, naive)
                if v > best * (1 + 1e-9):
                    best, idx, improved = v, trial, True
        if not improved:
            break
    return tuple(sorted(idx)), best


# ------------------------------------------------------------------ simple frequency rules
def freqs_loguniform(m):
    return tuple(np.unique(np.round(np.linspace(0, len(FGRID) - 1, m)).astype(int)))


def freqs_heuristic(ic, d, m, frac=0.1):
    """Engineering heuristic: m log-uniform frequencies inside the 'sensitive band' where the
    aggregate normalised sensitivity to the parameters of interest at the nominal medium
    exceeds `frac` of its maximum."""
    G = ic.G(d, M.THETA_C)
    score = np.sqrt(np.einsum("lii->l", G[:, :4, :4]))
    band = np.where(score >= frac * score.max())[0]
    lo, hi = band.min(), band.max()
    idx = np.unique(np.round(np.linspace(lo, hi, m)).astype(int))
    k = 0
    while len(idx) < m:       # degenerate narrow band: widen symmetrically
        k += 1
        idx = np.unique(np.round(np.linspace(max(lo - k, 0), min(hi + k, len(FGRID) - 1), m)).astype(int))
    return tuple(idx)


def freqs_greedy_dopt(ic, d, m, th=None):
    th = M.THETA_C if th is None else th
    G = ic.G(d, th); chosen = []
    for _ in range(m):
        best, bl = -np.inf, None
        for l in range(len(FGRID)):
            if l in chosen:
                continue
            v = subset_value([G], ic.P, chosen + [l], "D", ic.naive)
            if v > best:
                best, bl = v, l
        chosen.append(bl)
    return tuple(sorted(chosen))


# ------------------------------------------------------------------ scenarios
def corner_scenarios():
    pts = [M.THETA_C.copy()]
    for c in itertools.product([0, 1], repeat=4):
        pts.append(np.array([M.THETA_BOX[i, c[i]] for i in range(4)]))
    return pts


def sobol_pool(n, seed):
    s = qmc.Sobol(4, scramble=True, seed=seed).random(n)
    return M.THETA_BOX[:, 0] + s * (M.THETA_BOX[:, 1] - M.THETA_BOX[:, 0])


def worst_case(ic, d, idx, n_pool=256, seed=0, kind="E", refine=4, naive_eval=False):
    """Independent worst-case evaluation over Theta (Sobol pool + local refinement)."""
    wts = np.zeros(len(FGRID)); wts[list(idx)] = 1.0
    pool = list(sobol_pool(n_pool, seed)) + corner_scenarios()

    def f(th):
        th = np.clip(th, M.THETA_BOX[:, 0], M.THETA_BOX[:, 1])
        G = ic.G(d, th)
        A = np.tensordot(wts, G, axes=(0, 0))
        S = A[:4, :4] if naive_eval else M.efficient_info(A, ic.P)
        return crit(S, kind)

    vals = np.array([f(t) for t in pool])
    order = np.argsort(vals)[:refine]
    best_v, best_t = vals[order[0]], pool[order[0]]
    for o in order:
        r = minimize(lambda x: f(x), pool[o], method="Nelder-Mead",
                     options=dict(maxfev=120, xatol=1e-3, fatol=1e-6))
        th = np.clip(r.x, M.THETA_BOX[:, 0], M.THETA_BOX[:, 1])
        v = f(th)
        if v < best_v:
            best_v, best_t = v, th
    return best_v, best_t


def solve_frequencies(ic, d, m, scen, kind="E", rounding="swap"):
    Gs = [ic.G(d, t) for t in scen]
    w = design_weights(Gs, ic.P, m, kind, naive=ic.naive)
    idx = round_topm(w, m)
    if rounding == "swap":
        idx, _ = swap_refine(Gs, ic.P, idx, kind, ic.naive)
    return idx, w


def robust_frequency_design(ic, d, m, kind="E", seed=0, rounds=5, rounding="swap", scen0=None):
    """Exchange (cutting-plane) algorithm for the semi-infinite minimax problem."""
    scen = corner_scenarios() if scen0 is None else list(scen0)
    hist = []
    if kind == "A" or len(scen) == 1:
        idx, w = solve_frequencies(ic, d, m, scen, kind, rounding)
        return idx, w, scen, hist
    for r in range(rounds):
        idx, w = solve_frequencies(ic, d, m, scen, kind, rounding)
        cur = subset_value([ic.G(d, t) for t in scen], ic.P, idx, kind, ic.naive)
        wc, th = worst_case(ic, d, idx, seed=seed + 101 * r, kind=kind)
        hist.append((cur, wc))
        if wc >= cur * (1 - 1e-3):
            break
        scen.append(th)
    return idx, w, scen, hist


# ------------------------------------------------------------------ geometry search
def relaxed_value(Gs, P, wts, kind, naive):
    Gs = np.asarray(Gs)
    A = np.einsum("l,tlij->tij", wts, Gs)
    if naive:
        S = A[:, :4, :4]
    else:
        Att, Atn, Ann = A[:, :4, :4], A[:, :4, 4:], A[:, 4:, 4:] + P[None]
        S = Att - Atn @ np.linalg.solve(Ann, np.transpose(Atn, (0, 2, 1)))
    lam = np.linalg.eigvalsh(S)
    if kind == "A":
        return lam[:, 0].mean()
    return lam[:, 0].min() if kind == "E" else np.log(np.maximum(lam, 1e-300)).sum(1).min()


def geom_objective(ic, N, x, m, scen, kind, freq_rule):
    d = (N, x[0], x[1], x[2])
    if np.any(np.array(x) < GEOM_LB) or np.any(np.array(x) > GEOM_UB) or not feasible(d):
        return -1e30, None
    Gs = [ic.G(d, t) for t in scen]
    if freq_rule == "codesign":
        w = design_weights(Gs, ic.P, m, kind, naive=ic.naive, iters=120)
        return relaxed_value(Gs, ic.P, w, kind, ic.naive), w
    idx = freqs_heuristic(ic, d, m) if freq_rule == "heuristic" else freqs_loguniform(m)
    return subset_value(Gs, ic.P, idx, kind, ic.naive), idx


def _opt_N(args):
    op, sd_scale, naive, N, m, scen, kind, freq_rule, maxfev, readout = args
    M.NOISE["readout"] = readout
    ic = InfoCache(op, sd_scale, naive)
    x0 = np.array([50e-6, 50e-6, 3.0e-3])
    while not feasible((N, *x0)):
        x0[:2] *= 0.85
        if x0[0] < GEOM_LB[0] * 1.05:
            return (-np.inf, None)
    best = [-np.inf, None]
    def f(z):
        x = np.exp(z)
        v, _ = geom_objective(ic, N, x, m, scen, kind, freq_rule)
        if v > best[0]:
            best[0], best[1] = v, (N, *x)
        return -v if v > -1e29 else 1e30
    starts = [np.log(x0), np.log(np.array([x0[0] * 2, x0[1], x0[2]])) if feasible((N, x0[0] * 2, x0[1], x0[2])) else np.log(x0)]
    for z0 in starts:
        minimize(f, z0, method="Nelder-Mead", options=dict(maxfev=maxfev, xatol=2e-3, fatol=1e-7))
    return tuple(best)


def optimize_geometry(ic, m, scen, kind="E", freq_rule="codesign", n_list=N_RANGE, maxfev=70, pool=None):
    args = [(ic.op, None, ic.naive, N, m, scen, kind, freq_rule, maxfev, M.NOISE["readout"]) for N in n_list]
    res = pool.map(_opt_N, args) if pool is not None else list(map(_opt_N, args))
    res = [r for r in res if r[1] is not None]
    v, d = max(res, key=lambda r: r[0])
    return v, d, res


def codesign(ic, m, kind="E", seed=0, freq_rule="codesign", rounds=3, scen0=None, pool=None, verbose=False):
    """Full robust co-design: geometry search (integer N enumeration x bounded simplex search
    on the relaxed inner value), convex frequency design, and scenario exchange."""
    scen = corner_scenarios() if scen0 is None else list(scen0)
    log = []
    for r in range(rounds):
        v, d, allN = optimize_geometry(ic, m, scen, kind, freq_rule, pool=pool)
        if freq_rule == "codesign":
            idx, w, scen, hist = robust_frequency_design(ic, d, m, kind, seed + 7 * r, scen0=scen)
        elif freq_rule == "heuristic":
            idx = freqs_heuristic(ic, d, m)
        else:
            idx = freqs_loguniform(m)
        cur = subset_value([ic.G(d, t) for t in scen], ic.P, idx, kind, ic.naive)
        if kind == "A" or len(scen) == 1 or rounds == 1:
            log.append(dict(round=r, d=d, idx=idx, cur=cur, wc=None, nscen=len(scen), allN=allN))
            break
        wc, th = worst_case(ic, d, idx, seed=seed + 13 * r, kind=kind)
        log.append(dict(round=r, d=d, idx=idx, cur=cur, wc=wc, nscen=len(scen), allN=allN))
        if verbose:
            print("round", r, np.round(np.array(d) * [1, 1e6, 1e6, 1e3], 2), idx, round(cur, 5), round(wc, 5), flush=True)
        if wc >= cur * (1 - 1e-3):
            break
        scen.append(th)
    return d, idx, scen, log
