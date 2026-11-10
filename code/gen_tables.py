"""Generate LaTeX tables and number macros directly from the result files."""
import json, numpy as np
import model as M

ev = {(r["name"], r["m"]): r for r in json.load(open("../results/eval_fim.json"))}
mc = json.load(open("../results/mc_main.json"))
rd = json.load(open("../results/rounding.json"))
rs = json.load(open("../results/real_summary.json"))
try:
    mci = json.load(open("../results/mc_ideal.json"))
except FileNotFoundError:
    mci = {}
TOL = M.THETA_TOL

def crlb_nat(r):
    v = np.mean(r["ratio"], 0) * TOL
    return [100 * v[0], 100 * v[1], 100 * v[2], v[3]]

def geom(r):
    N, w, s, D = r["d"]
    return "%d/%d/%d/%.2f" % (N, round(w * 1e6), round(s * 1e6), D * 1e3)

OPN = {"single": "S", "diff": "D"}
def op(r):
    return OPN[r["op"]] + ("" if r["op"] == "single" else ("-a" if r["ro"] == "analog" else "-d"))

FR = {"U": "log-unif.", "H": "heuristic", "R": "robust", "G": "greedy-D"}
rows = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B3-dig", "Ours-dig", "Ours"]
lab = {"B3-dig": "B3$'$", "Ours-dig": "Prop.$'$", "Ours": "\\textbf{Proposed}"}
lines = []
for n in rows:
    r = ev[(n, 6)]
    c = crlb_nat(r)
    g = "co-des." if r["geom"] == "opt" else "template"
    m = mc.get(n)
    mcs = "%.0f & %.2f" % (100 * np.mean([x["compliance"] for x in m]), np.mean([x["worst_p95"] for x in m])) if m else "-- & --"
    if n == "B4":
        mcs = "\\multicolumn{2}{c}{(= B3)}"
    lines.append("%s & %s & %s & %s & %s & %.4f & %.1f & %.1f & %.1f & %.3f & %s \\\\" % (
        lab.get(n, n), op(r), g, geom(r), FR[r["freq"]], np.mean(r["ME"]), c[0], c[1], c[2], c[3], mcs))
    if n in ("B7", "B4"):
        lines.append("\\midrule")
open("../tables/tab_main_rows.tex", "w").write("\n".join(lines) + "\n")

abl = [("Ours", "Robust E, exchange (proposed)"), ("A-noexch", "Corner scenarios only (no exchange)"),
       ("A-average", "Average-case E over 32 Sobol points"), ("A-Dopt", "Robust D-criterion"),
       ("A-nominal", "Nominal E at center of $\\Theta$"), ("A-naive", "Robust E, nuisance ignored ($\\mathbf A_{\\theta\\theta}$)"),
       ("A-greedyD", "Template geometry, greedy D at center")]
L = []
for n, t in abl:
    r = ev[(n, 6)]; c = crlb_nat(r)
    L.append("%s & %s & %.4f & %.1f & %.1f & %.1f & %.3f \\\\" % (t, geom(r), np.mean(r["ME"]), c[0], c[1], c[2], c[3]))
open("../tables/tab_abl_rows.tex", "w").write("\n".join(L) + "\n")

R = []
for m in ["4", "6", "8", "10"]:
    x = rd[m]
    ex = "%.4f" % x["coarse_exhaustive"] if "coarse_exhaustive" in x else "--"
    exs = "%.4f" % x["coarse_swap"] if "coarse_swap" in x else "--"
    R.append("%s & %.4f & %.4f & %.4f & %.1f & %s & %s \\\\" % (m, x["relaxed"], x["topm"], x["swap"], 100 * x["gap_swap"], exs, ex))
open("../tables/tab_round_rows.tex", "w").write("\n".join(R) + "\n")

# ---------------- number macros
N = {}
def put(k, v): N[k] = v
me = lambda n, m=6: np.mean(ev[(n, m)]["ME"])
put("MEoursSix", "%.3f" % me("Ours")); put("MEbOneSix", "%.4f" % me("B1"))
put("MEbSevenSix", "%.3f" % me("B7")); put("MEbSixSix", "%.3f" % me("B6")); put("MEbFiveSix", "%.3f" % me("B5"))
put("MEbThreeSix", "%.3f" % me("B3")); put("MEdigSix", "%.3f" % me("Ours-dig")); put("MEbThreeDigSix", "%.4f" % me("B3-dig"))
put("gainBOne", "%.0f" % (me("Ours") / me("B1"))); put("gainBest", "%.2f" % (me("Ours") / max(me(n) for n in ["B5", "B6", "B7"])))
for m in [4, 8, 10]:
    put("MEours%s" % {4: "Four", 8: "Eight", 10: "Ten"}[m], "%.3f" % me("Ours", m))
put("gainFour", "%.2f" % (me("Ours", 4) / max(me(n, 4) for n in ["B5", "B6", "B7"])))
put("gainTen", "%.2f" % (me("Ours", 10) / max(me(n, 10) for n in ["B5", "B6", "B7"])))
put("mFourOursVsBOneTen", "%.1f" % (me("Ours", 4) / me("B1", 10)))
c = crlb_nat(ev[("Ours", 6)])
put("crlbSigOurs", "%.1f" % c[0]); put("crlbKapOurs", "%.1f" % c[1]); put("crlbTauOurs", "%.1f" % c[2]); put("crlbAlpOurs", "%.3f" % c[3])
c1 = crlb_nat(ev[("B1", 6)])
put("crlbSigBOne", "%.1f" % c1[0]); put("crlbTauBOne", "%.1f" % c1[2])
dO = ev[("Ours", 6)]["d"]
put("oursN", "%d" % dO[0]); put("oursW", "%d" % round(dO[1] * 1e6)); put("oursS", "%d" % round(dO[2] * 1e6)); put("oursD", "%.2f" % (dO[3] * 1e3))
put("oursFreqs", ", ".join(("%.0f~kHz" % (f / 1e3)) if f < 1e6 else ("%.2f~MHz" % (f / 1e6)) for f in ev[("Ours", 6)]["freqs"]))
put("nominalRatio", "%.1f" % (me("Ours") / me("A-nominal"))); put("naiveRatio", "%.0f" % (me("Ours") / me("A-naive")))
put("avgLoss", "%.0f" % (100 * (1 - me("A-average") / me("Ours")))); put("doptLoss", "%.0f" % (100 * (1 - me("A-Dopt") / me("Ours"))))
for n, k in [("Ours", "Ours"), ("Ours-dig", "Dig"), ("B1", "BOne"), ("B3", "BThree"), ("B5", "BFive"), ("B6", "BSix"), ("B7", "BSeven"), ("B3-dig", "BThreeDig"), ("B2", "BTwo")]:
    if n in mc:
        put("comp" + k, "%.0f" % (100 * np.mean([x["compliance"] for x in mc[n]])))
        put("ptail" + k, "%.2f" % np.mean([x["worst_p95"] for x in mc[n]]))
        put("wrmse" + k, "%.2f" % np.mean([x["worst_rmse"] for x in mc[n]]))
put("gapSwapSix", "%.1f" % (100 * rd["6"]["gap_swap"])); put("gapTopSix", "%.0f" % (100 * rd["6"]["gap_topm"]))
put("gapSwapTen", "%.1f" % (100 * rd["10"]["gap_swap"])); put("gapSwapFour", "%.0f" % (100 * rd["4"]["gap_swap"]))
secs = [r["secs"] for r in json.load(open("../results/designs.json")) if r["name"] == "Ours"]
put("timeMin", "%.0f" % min(secs)); put("timeMax", "%.0f" % max(secs))
# real data
put("realNoise", "%.2f" % (100 * 0.0016)); put("realNoiseMax", "%.2f" % (100 * rs["noise_rel_median"]))
put("realFit", "%.1f" % (100 * rs["fit_rel_rms_median"])); put("realFitNinety", "%.1f" % (100 * rs["fit_rel_rms_p90"]))
put("realCV", "%.1f" % (100 * rs["channel_cv_G_median"])); put("realCVNinety", "%.1f" % (100 * rs["channel_cv_G_p90"]))
for m in [4, 5, 6]:
    w = {4: "Four", 5: "Five", 6: "Six"}[m]
    put("rRob" + w, "%.4f" % rs[f"robust_{m}"]["rmse"]); put("rLog" + w, "%.4f" % rs[f"log-uniform_{m}"]["rmse"])
    put("rNom" + w, "%.4f" % rs[f"nominal_{m}"]["rmse"])
    put("rRobWc" + w, "%.4f" % rs[f"robust_{m}"]["wc"]); put("rLogWc" + w, "%.4f" % rs[f"log-uniform_{m}"]["wc"]); put("rNomWc" + w, "%.4f" % rs[f"nominal_{m}"]["wc"])
    d = rs[f"diff_robust_minus_logu_{m}"]; put("rDiffCI" + w, "[%.4f, %.4f]" % (d[1], d[2]))
    put("rOpSingle" + w, "%.4f" % rs[f"op_single_{m}"]); put("rOpMatched" + w, "%.3f" % rs[f"op_matched-ref_{m}"]); put("rOpMis" + w, "%.3f" % rs[f"op_mismatched-ref_{m}"])
    put("rConcRob" + w, "%.1f" % (100 * rs[f"conc_robust_{m}"])); put("rConcLog" + w, "%.1f" % (100 * rs[f"conc_log-uniform_{m}"]))
put("rConcFull", "%.1f" % (100 * rs["conc_full"]))
put("rSpearman", "%.1f" % rs["spearman"]); put("rSpearmanP", "%.2f" % rs["spearman_p"])
put("rRsChange", "%.1f" % (100 * rs["Rs_rel_change_median"])); put("rRsChangeNinety", "%.0f" % (100 * rs["Rs_rel_change_p90"]))
for n, k in [("Ours", "Ours"), ("Ours-dig", "Dig"), ("B7", "BSeven"), ("B3", "BThree"), ("B6", "BSix")]:
    if n in mci:
        put("idealComp" + k, "%.0f" % (100 * np.mean([x["compliance"] for x in mci[n]])))
        put("idealP" + k, "%.2f" % np.mean([x["worst_p95"] for x in mci[n]]))
wc = json.load(open("../results/mc_worstcorner.json"))
for n, k in [("Ours", "Ours"), ("Ours-dig", "Dig"), ("B1", "BOne"), ("B3", "BThree"), ("B7", "BSeven"), ("B6@common", "BSix"), ("B5@common", "BFive")]:
    put("corner" + k, "%.0f" % (100 * wc[n]["compliance"]))
open("../tables/numbers.tex", "w").write("\n".join("\\newcommand{\\%s}{%s}" % (k, v) for k, v in N.items()) + "\n")
print(len(N), "macros")

# ---------------- real-data table rows (values x 10^3)
k = lambda v: "%.1f" % (1000 * v)
rows = []
for lab_, key in [("Log-uniform", "log-uniform"), ("Nominal (1~mM)", "nominal"), ("Robust (worst conc.)", "robust")]:
    cells = []
    for m in [4, 5, 6]:
        x = rs[f"{key}_{m}"]; c = "%s\\,/\\,%s" % (k(x["rmse"]), k(x["wc"]))
        if key == "robust" and m in (5, 6):
            c = "\\textbf{%s}" % c
        cells.append(c)
    rows.append(lab_ + " & " + " & ".join(cells) + " \\\\")
rows.append("$\\Delta$ robust$-$log-unif. & " + " & ".join("[%s, %s]" % (k(rs[f"diff_robust_minus_logu_{m}"][1]), k(rs[f"diff_robust_minus_logu_{m}"][2])) for m in [4, 5, 6]) + " \\\\")
rows.append("\\midrule")
rows.append("Matched reference & " + " & ".join(k(rs[f"op_matched-ref_{m}"]) for m in [4, 5, 6]) + " \\\\")
rows.append("Mismatched reference & " + " & ".join(k(rs[f"op_mismatched-ref_{m}"]) for m in [4, 5, 6]) + " \\\\")
rows.append("\\midrule")
rows.append("Conc. error (\\%), rob./log-u. & " + " & ".join("%.1f\\,/\\,%.1f" % (100 * rs[f"conc_robust_{m}"], 100 * rs[f"conc_log-uniform_{m}"]) for m in [4, 5, 6]) + " \\\\")
open("../tables/tab_real_rows.tex", "w").write("\n".join(rows) + "\n")
