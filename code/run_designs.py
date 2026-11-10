"""Compute all designs (baselines, proposed, ablations) and store them."""
import json, time, sys, os
import numpy as np, multiprocessing as mp
import model as M, design as D

OUT = "../results/designs.json"


def spec_list():
    S = []
    for m in [4, 6, 8, 10]:
        S += [
            dict(name="B1", m=m, op="single", ro="digital", geom="d0", freq="U"),
            dict(name="B2", m=m, op="single", ro="digital", geom="d0", freq="H"),
            dict(name="B3", m=m, op="diff", ro="analog", geom="d0", freq="U"),
            dict(name="B4", m=m, op="diff", ro="analog", geom="d0", freq="H"),
            dict(name="B5", m=m, op="diff", ro="analog", geom="d0", freq="R"),
            dict(name="B6", m=m, op="diff", ro="analog", geom="opt", freq="H"),
            dict(name="B7", m=m, op="single", ro="digital", geom="opt", freq="R"),
            dict(name="Ours", m=m, op="diff", ro="analog", geom="opt", freq="R"),
            dict(name="B3-dig", m=m, op="diff", ro="digital", geom="d0", freq="U"),
            dict(name="Ours-dig", m=m, op="diff", ro="digital", geom="opt", freq="R"),
        ]
    m = 6
    S += [
        dict(name="A-nominal", m=m, op="diff", ro="analog", geom="opt", freq="R", scen="nominal"),
        dict(name="A-average", m=m, op="diff", ro="analog", geom="opt", freq="R", kind="A", scen="sobol32"),
        dict(name="A-Dopt", m=m, op="diff", ro="analog", geom="opt", freq="R", kind="D"),
        dict(name="A-naive", m=m, op="diff", ro="analog", geom="opt", freq="R", naive=True),
        dict(name="A-noexch", m=m, op="diff", ro="analog", geom="opt", freq="R", rounds=1),
        dict(name="A-greedyD", m=m, op="diff", ro="analog", geom="d0", freq="G"),
    ]
    return S


def run_one(s, pool):
    M.NOISE["readout"] = s["ro"]
    ic = D.InfoCache(s["op"], naive=s.get("naive", False))
    kind = s.get("kind", "E")
    scen0 = None
    if s.get("scen") == "nominal":
        scen0 = [M.THETA_C.copy()]
    elif s.get("scen") == "sobol32":
        scen0 = list(D.sobol_pool(32, 99))
    t = time.time()
    m = s["m"]
    if s["geom"] == "d0":
        d = D.D0
        if s["freq"] == "U":
            idx = D.freqs_loguniform(m)
        elif s["freq"] == "H":
            idx = D.freqs_heuristic(ic, d, m)
        elif s["freq"] == "G":
            idx = D.freqs_greedy_dopt(ic, d, m)
        else:
            idx, _, _, _ = D.robust_frequency_design(ic, d, m, kind, seed=0, scen0=scen0)
    else:
        rule = "codesign" if s["freq"] == "R" else "heuristic"
        d, idx, scen, log = D.codesign(ic, m, kind, seed=0, freq_rule=rule,
                                       rounds=s.get("rounds", 3), scen0=scen0, pool=pool)
    out = dict(s)
    out.update(d=[float(x) for x in d], idx=[int(i) for i in idx],
               freqs=[float(D.FGRID[i]) for i in idx], secs=time.time() - t)
    return out


if __name__ == "__main__":
    pool = mp.Pool(2)
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    have = {(r["name"], r["m"]) for r in done}
    for s in spec_list():
        if (s["name"], s["m"]) in have:
            continue
        r = run_one(s, pool)
        done.append(r)
        json.dump(done, open(OUT, "w"), indent=1)
        print(r["name"], r["m"], np.round(np.array(r["d"]) * [1, 1e6, 1e6, 1e3], 1), r["idx"], round(r["secs"], 1), flush=True)
