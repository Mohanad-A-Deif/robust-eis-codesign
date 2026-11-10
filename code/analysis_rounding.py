"""Rounding certificate and exhaustive check of the frequency subset at the co-designed geometry."""
import json, itertools, time
import numpy as np
import model as M, design as D

des = {(r["name"], r["m"]): r for r in json.load(open("../results/designs.json"))}
out = {}
M.NOISE["readout"] = "analog"
ic = D.InfoCache("diff")
for m in [4, 6, 8, 10]:
    d = tuple(des[("Ours", m)]["d"])
    scen = D.corner_scenarios()
    Gs = np.array([ic.G(d, t) for t in scen])
    t0 = time.time()
    w = D.design_weights(list(Gs), ic.P, m, iters=600, rtol=0)
    t_relax = time.time() - t0
    relaxed = D.relaxed_value(Gs, ic.P, w, "E", False)
    top = D.round_topm(w, m)
    v_top = D.subset_value(Gs, ic.P, top)
    sw, v_sw = D.swap_refine(list(Gs), ic.P, top)
    rec = dict(m=m, relaxed=float(relaxed), topm=float(v_top), swap=float(v_sw), t_relax=t_relax,
               gap_topm=float(1 - v_top / relaxed), gap_swap=float(1 - v_sw / relaxed))
    if m in (4, 6):
        # exhaustive search on a coarse 16-point sub-grid (every 4th candidate frequency)
        sub = list(range(0, 64, 4)) + [63]
        sub = sorted(set(sub))
        Gc = Gs[:, sub]
        best = -np.inf
        for comb in itertools.combinations(range(len(sub)), m):
            v = D.subset_value(Gc, ic.P, comb)
            if v > best:
                best = v
        wc = D.design_weights(list(Gc), ic.P, m, iters=600, rtol=0)
        tc = D.round_topm(wc, m); sc, vsc = D.swap_refine(list(Gc), ic.P, tc)
        rec.update(coarse_exhaustive=float(best), coarse_relaxed=float(D.relaxed_value(Gc, ic.P, wc, "E", False)),
                   coarse_swap=float(vsc), n_coarse=len(sub))
    out[m] = rec
    print(rec, flush=True)
json.dump(out, open("../results/rounding.json", "w"), indent=1)
