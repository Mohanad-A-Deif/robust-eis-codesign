import json, pickle, numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import figstyle as S, realdata as R
from conc_eval import conc_errors
S.setup(); C = S.C

o = json.load(open("../results/realdata_results.json")); rows = o["rows"]
fits = pickle.load(open("../results/gapfits.pkl", "rb"))
GAPS = ["80", "120", "160", "200", "240"]; CONC = [0.1, 0.5, 1.0, 2.0, 4.0]
DN = ["log-uniform", "heuristic", "nominal", "robust"]
rng = np.random.default_rng(0)


def dev(r): return r["lnG"] - r["lnG_full"]


def sel(**k): return [r for r in rows if all(r[a] == b for a, b in k.items())]


def worst_conc_rmse(rs):
    return max(np.sqrt(np.mean([dev(r) ** 2 for r in rs if r["c"] == c])) for c in CONC)


summary = {}
for m in [4, 5, 6]:
    for dn in DN:
        rs = sel(m=m, design=dn, op="single")
        e = np.array([dev(r) for r in rs])
        summary[f"{dn}_{m}"] = dict(rmse=float(np.sqrt(np.mean(e ** 2))), wc=float(worst_conc_rmse(rs)),
                                    p95=float(np.percentile(np.abs(e), 95)))
    # paired bootstrap robust vs log-uniform (resample spectra)
    a = {(r["gap"], r["ch"], r["c"]): dev(r) for r in sel(m=m, design="robust", op="single")}
    b = {(r["gap"], r["ch"], r["c"]): dev(r) for r in sel(m=m, design="log-uniform", op="single")}
    keys = sorted(a); A = np.array([a[k] for k in keys]); B = np.array([b[k] for k in keys])
    bs = []
    for _ in range(4000):
        i = rng.integers(0, len(keys), len(keys))
        bs.append(np.sqrt(np.mean(A[i] ** 2)) - np.sqrt(np.mean(B[i] ** 2)))
    summary[f"diff_robust_minus_logu_{m}"] = [float(np.mean(bs)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    for op in ["single", "matched-ref", "mismatched-ref"]:
        rs = [r for r in sel(m=m, design="robust", op=op) if r["c"] != 1.0]
        summary[f"op_{op}_{m}"] = float(np.sqrt(np.mean([dev(r) ** 2 for r in rs])))
    for dn in DN:
        E = conc_errors(rows, lambda r, dn=dn, m=m: r["m"] == m and r["design"] == dn and r["op"] == "single")
        summary[f"conc_{dn}_{m}"] = float(np.sqrt(np.mean([x["e"] ** 2 for x in E])))
full = [dict(r, lnG=r["lnG_full"]) for r in sel(m=6, design="robust", op="single")]
summary["conc_full"] = float(np.sqrt(np.mean([x["e"] ** 2 for x in conc_errors(full, lambda r: True)])))
pv, emp = [], []
for g in GAPS:
    rs = sel(m=6, design="robust", op="single", gap=g)
    pv.append(float(np.sqrt(np.mean([r["pred_wc_var"] for r in rs])))); emp.append(float(np.sqrt(np.mean([dev(r) ** 2 for r in rs]))))
sp = spearmanr(pv, emp)
summary.update(pred_gap=pv, emp_gap=emp, spearman=float(sp.statistic), spearman_p=float(sp.pvalue))
# common-mode check: series resistance change between reference (1 mM) and other concentrations on the same channel
dR = []
for r in fits:
    if r["c"] in CONC and r["c"] != 1.0:
        ref = [q for q in fits if q["geom"] == r["geom"] and q["ch"] == r["ch"] and q["c"] == 1.0][0]
        dR.append(abs(np.exp(r["xfull"][4]) - np.exp(ref["xfull"][4])) / np.exp(ref["xfull"][4]))
summary["Rs_rel_change_median"] = float(np.median(dR)); summary["Rs_rel_change_p90"] = float(np.percentile(dR, 90))
summary.update({k: o[k] for k in o if k not in ("rows", "freq")})
json.dump(summary, open("../results/real_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.66), gridspec_kw=dict(wspace=0.62, width_ratios=[1.15, 1, 1, 1]))
for g, col, mk in [("80", C["blue"], "o"), ("240", C["red"], "s")]:
    r = [q for q in fits if q["geom"] == g and q["c"] == 1.0][0]
    ax[0].loglog(r["f"], np.abs(r["Z"]), mk, ms=1.6, color=col, mfc="none", mew=0.4, label="%s $\\mu$m meas." % g)
    ax[0].loglog(r["f"], np.abs(R.zmodel(r["xfull"], r["f"])), "-", color=C["black"], lw=0.7)
ax[0].plot([], [], "-", color=C["black"], lw=0.7, label="model fit")
ax[0].set_xlabel("Frequency (Hz)"); ax[0].set_ylabel("$|Z|$ ($\\Omega$)"); ax[0].legend(fontsize=5.2, loc="lower left")
mk = {"log-uniform": "o", "heuristic": "s", "nominal": "^", "robust": "*"}
cl = {"log-uniform": C["gray"], "heuristic": C["brown"], "nominal": C["blue"], "robust": C["red"]}
for dn in ["log-uniform", "nominal", "robust"]:
    ax[1].plot([4, 5, 6], [summary[f"{dn}_{m}"]["wc"] for m in [4, 5, 6]], marker=mk[dn], color=cl[dn],
               mfc="white" if dn != "robust" else cl[dn], ms=4, label=("log-uniform = heuristic" if dn == "log-uniform" else dn))
ax[1].set_xlabel("$m$"); ax[1].set_ylabel("Worst-conc. RMS dev. of $\\ln G$"); ax[1].set_xticks([4, 5, 6])
ax[1].legend(fontsize=5.0, loc="upper right", handlelength=1.4)
ops = ["single", "matched-ref", "mismatched-ref"]; hat = ["", "////", "xxxx"]
for i, op in enumerate(ops):
    ax[2].bar(np.arange(3) + (i - 1) * 0.25, [summary[f"op_{op}_{m}"] for m in [4, 5, 6]], width=0.25,
              color=C["red"] if op == "single" else "white", edgecolor=C["black"], hatch=hat[i], lw=0.6, label=op)
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["4", "5", "6"]); ax[2].set_xlabel("$m$")
ax[2].set_ylabel("RMS dev. of $\\ln G$"); ax[2].set_ylim(0, 0.085); ax[2].legend(fontsize=5.0, loc="upper left", ncol=1, handlelength=1.4)
for g, x, y in zip(GAPS, pv, emp):
    ax[3].plot(x, y, "o", color=C["blue"], ms=3.5)
    ax[3].annotate(g + " $\\mu$m", (x, y), textcoords="offset points", xytext={"240": (-4, 5), "200": (3, 4), "160": (4, -4)}.get(g, (-22, -1) if g == "80" else (3, 2)), fontsize=5.2)
ax[3].set_xlabel("Predicted worst-case CRLB"); ax[3].set_ylabel("Observed RMS dev.")
for a, t in zip(ax, "abcd"):
    a.text(-0.32, 1.03, "(%s)" % t, transform=a.transAxes, fontsize=8)
fig.savefig("../figs/Fig6_real.png"); fig.savefig("../figs/Fig6_real.pdf")
