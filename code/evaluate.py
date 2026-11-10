"""Worst-case information evaluation of every design over Theta (independent of the design
scenarios): Sobol pool + local refinement, 5 evaluation seeds."""
import json, numpy as np
import model as M, design as D


def per_param_worst(ic, d, idx, seed, n=256):
    wts = np.zeros(len(D.FGRID)); wts[list(idx)] = 1.0
    worst = np.zeros(4); dvals = []
    for th in list(D.sobol_pool(n, seed)) + D.corner_scenarios():
        A = np.tensordot(wts, ic.G(d, th), axes=(0, 0))
        I = M.efficient_info(A, ic.P)
        worst = np.maximum(worst, np.sqrt(np.diag(np.linalg.inv(I))))
        dvals.append(np.linalg.slogdet(I)[1])
    return worst, min(dvals)


if __name__ == "__main__":
    designs = json.load(open("../results/designs.json"))
    out = []
    for r in designs:
        M.NOISE["readout"] = r["ro"]
        ic = D.InfoCache(r["op"])           # evaluation always with the nuisance-marginalised FIM
        d = tuple(r["d"]); idx = r["idx"]
        E, PP, DD = [], [], []
        for s in range(5):
            v, th = D.worst_case(ic, d, idx, n_pool=256, seed=1000 + s, kind="E")
            w, dmin = per_param_worst(ic, d, idx, 1000 + s)
            E.append(v); PP.append(w.tolist()); DD.append(dmin)
        rr = dict(r); rr.update(ME=E, ratio=PP, MD=DD)
        out.append(rr)
        print(r["name"], r["m"], np.mean(E), np.std(E), np.round(np.mean(PP, 0), 2), flush=True)
        json.dump(out, open("../results/eval_fim.json", "w"), indent=1)
