import json, numpy as np
import model as M, design as D, estimate as E
des = {(r["name"], r["m"]): r for r in json.load(open("../results/designs.json"))}
out = {}
for nm in ["B1", "B3", "B6", "B7", "Ours-dig", "Ours"]:
    r = des[(nm, 6)]; M.NOISE["readout"] = r["ro"]
    ic = D.InfoCache(r["op"])
    v, th = D.worst_case(ic, tuple(r["d"]), r["idx"], seed=1000)
    Er = E.run_mc_fixed(r, th, n=60)
    s = E.summarize(Er)
    wts = np.zeros(64); wts[r["idx"]] = 1
    I = M.efficient_info(np.tensordot(wts, ic.G(tuple(r["d"]), th), axes=(0, 0)), ic.P)
    s.update(theta=th.tolist(), crlb_ratio=np.sqrt(np.diag(np.linalg.inv(I))).tolist(), lam=float(v),
             emp_sd=np.std(Er, 0).tolist())
    out[nm] = s
    print(nm, np.round(th, 2), "crlb", np.round(s["crlb_ratio"], 2), "emp rmse", np.round(s["rmse"], 2), "comp", s["compliance"], flush=True)
json.dump(out, open("../results/mc_worstcorner.json", "w"), indent=1)
