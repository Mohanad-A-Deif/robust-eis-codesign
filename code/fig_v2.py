"""Redesigned result figures (Figs. 2-4) with a consistent method style and shared legends."""
import json, pickle, numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as S, realdata as R
S.setup(); C = S.C
plt.rcParams.update({"axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.4, "axes.axisbelow": True})

ST = {  # consistent style for every method
    "B1": dict(c="#6E6E6E", m="o", f="white", ls=(0, (4, 2)), lab="B1"),
    "B2": dict(c="#6E6E6E", m="s", f="white", ls=":", lab="B2"),
    "B3": dict(c=C["brown"], m="^", f="white", ls="--", lab="B3 (=B4)"),
    "B5": dict(c=C["blue"], m="D", f="white", ls="-.", lab="B5"),
    "B6": dict(c=C["green"], m="P", f=C["green"], ls="-.", lab="B6"),
    "B7": dict(c=C["purple"], m="X", f=C["purple"], ls="--", lab="B7"),
    "B3-dig": dict(c=C["brown"], m="v", f=C["brown"], ls=":", lab="B3$'$"),
    "Ours-dig": dict(c=C["red"], m="*", f="white", ls=":", lab="Proposed$'$ (digital)"),
    "Ours": dict(c=C["red"], m="*", f=C["red"], ls="-", lab="Proposed (analog)"),
}
def mstyle(n, ms=5):
    s = ST[n]; size = ms + 2.5 if s["m"] == "*" else ms
    return dict(color=s["c"], marker=s["m"], mfc=s["f"], mec=s["c"], ms=size, mew=0.9)
def handles(names, line=True, ms=5):
    return [Line2D([], [], ls=ST[n]["ls"] if line else "none", lw=1.1, **mstyle(n, ms)) for n in names], [ST[n]["lab"] for n in names]
def tag(ax, t, x=-0.16):
    ax.text(x, 1.04, "(%s)" % t, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
def dotplot(ax, names, vals, err=None, ref=None, logy=False, ms=4.5):
    """vals[n] -> array over groups; markers offset within each group, thin stems from the baseline."""
    G = len(next(iter(vals.values()))); k = len(names); width = 0.78
    offs = (np.arange(k) - (k - 1) / 2) * width / k
    base = ref if ref is not None else 0
    for g in range(G):
        if g % 2 == 0:
            ax.axvspan(g - 0.5, g + 0.5, color="#F4F4F4", zorder=0, lw=0)
    for i, n in enumerate(names):
        x = np.arange(G) + offs[i]; y = np.asarray(vals[n])
        ax.vlines(x, base, y, color=ST[n]["c"], lw=0.8, alpha=0.8, zorder=2)
        if err is not None:
            ax.errorbar(x, y, yerr=err[n], fmt="none", ecolor=ST[n]["c"], elinewidth=0.7, capsize=1.5, zorder=3)
        ax.plot(x, y, ls="none", zorder=4, **mstyle(n, ms))
    if ref is not None:
        ax.axhline(ref, color=C["black"], ls="--", lw=0.8, zorder=1)
    ax.set_xlim(-0.5, G - 0.5); ax.grid(axis="x", visible=False)
    if logy:
        ax.set_yscale("log")


# =============================================================== Fig. 2
ev = {(r["name"], r["m"]): r for r in json.load(open("../results/eval_fim.json"))}
names = ["B1", "B2", "B3", "B5", "B6", "B7", "Ours-dig", "Ours"]
fig = plt.figure(figsize=(7.16, 2.15))
a1 = fig.add_axes([0.075, 0.17, 0.36, 0.62]); a2 = fig.add_axes([0.53, 0.17, 0.46, 0.62])
ms_ = [4, 6, 8, 10]
for n in names:
    y = [np.mean(ev[(n, m)]["ME"]) for m in ms_]
    a1.semilogy(ms_, y, ls=ST[n]["ls"], lw=1.1 if n != "Ours" else 1.6, zorder=5 if n == "Ours" else 3, **mstyle(n))
a1.set_xticks(ms_); a1.set_xlabel("Frequency budget $m$"); a1.set_ylabel("Worst-case $\\lambda_{\\min}(\\mathbf{I})$")
a1.set_xlim(3.6, 10.4)
yb, yo = np.mean(ev[("B1", 6)]["ME"]), np.mean(ev[("Ours", 6)]["ME"])
a1.annotate("", xy=(6.35, yb), xytext=(6.35, yo), arrowprops=dict(arrowstyle="<->", lw=0.7, color=C["black"]))
a1.text(6.5, np.sqrt(yb * yo), "$\\times$%.0f" % (yo / yb), fontsize=6.5, va="center")
pn = ["$\\sigma_0$", "$\\kappa$", "$\\tau$", "$\\alpha$"]
sub = ["B1", "B3", "B5", "B6", "B7", "Ours-dig", "Ours"]
dotplot(a2, sub, {n: np.mean(ev[(n, 6)]["ratio"], 0) for n in sub}, ref=1.0, logy=True)
a2.set_xticks(range(4)); a2.set_xticklabels(pn, fontsize=8)
a2.set_ylabel("Worst-case CRLB / target", labelpad=1); a2.set_ylim(0.6, 20)
a2.text(3.45, 0.95, "target", fontsize=5.8, ha="right", va="top")
tag(a1, "a", -0.2); tag(a2, "b", -0.13)
h, l = handles(names)
fig.legend(h, l, ncol=8, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6.2, columnspacing=1.0, handlelength=2.2, frameon=False)
fig.savefig("../figs/Fig2_fim.png", dpi=600); fig.savefig("../figs/Fig2_fim.pdf"); plt.close(fig)

# =============================================================== Fig. 3
mc = json.load(open("../results/mc_main.json")); st = json.load(open("../results/mc_stress.json"))
order = ["B1", "B2", "B3", "B5", "B6", "B7", "B3-dig", "Ours-dig", "Ours"]
fig = plt.figure(figsize=(7.16, 1.95))
ax = [fig.add_axes([0.06, 0.2, 0.21, 0.6]), fig.add_axes([0.33, 0.2, 0.25, 0.6]),
      fig.add_axes([0.645, 0.2, 0.155, 0.6]), fig.add_axes([0.84, 0.2, 0.155, 0.6])]
v = np.array([100 * np.mean([x["compliance"] for x in mc[n]]) for n in order])
e = np.array([100 * np.std([x["compliance"] for x in mc[n]]) for n in order])
for i, n in enumerate(order):
    ax[0].barh(i, v[i], color=ST[n]["c"], alpha=0.18 if n != "Ours" else 0.9, height=0.62, zorder=2)
    ax[0].errorbar(v[i], i, xerr=e[i], fmt="none", ecolor=C["black"], elinewidth=0.6, capsize=1.5, zorder=3)
    ax[0].plot(v[i], i, ls="none", zorder=4, **mstyle(n, 4.5))
    ax[0].text(min(v[i] + e[i] + 2, 101), i, "%.0f" % v[i], va="center", fontsize=5.6)
ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels([ST[n]["lab"].replace(" (analog)", "").replace(" (digital)", "").replace(" (=B4)", "") for n in order], fontsize=6)
ax[0].set_xlim(0, 112); ax[0].set_xlabel("Compliance (%)"); ax[0].grid(axis="y", visible=False)
ax[0].invert_yaxis()
sub = ["B1", "B3", "B5", "B6", "B7", "Ours-dig", "Ours"]
dotplot(ax[1], sub, {n: np.mean([x["p95"] for x in mc[n]], 0) for n in sub}, ref=2.0)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(pn, fontsize=8); ax[1].set_ylabel("P95 of $|e_k|/t_k$"); ax[1].set_ylim(0, 7.6)
ax[1].text(3.45, 2.12, "$\\pm2$ band", fontsize=5.6, ha="right", va="bottom")
amps = [1.0, 1.5, 2.0, 3.0]
for n in ["B3", "B6", "B7", "Ours"]:
    ax[2].plot(amps, [np.mean([x["worst_p95"] for x in st[n]["amp%s" % a]]) for a in amps], ls=ST[n]["ls"],
               lw=1.1 if n != "Ours" else 1.6, **mstyle(n, 4))
ax[2].set_xlabel("Amplification $\\gamma$"); ax[2].set_ylabel("Worst-param. P95"); ax[2].set_xticks(amps); ax[2].set_xticklabels(["1", "1.5", "2", "3"])
rbs = [0.0, 0.01, 0.02, 0.03]
for n in ["B3", "B6", "Ours"]:
    ax[3].plot(100 * np.array(rbs), [np.mean([x["worst_p95"] for x in st[n]["ref%s" % r]]) for r in rbs], ls=ST[n]["ls"],
               lw=1.1 if n != "Ours" else 1.6, **mstyle(n, 4))
ax[3].set_xlabel("Ref. error s.d. (%)"); ax[3].set_xticks([0, 1, 2, 3]); ax[3].set_ylim(2, 4.3)
for a, t, x in zip(ax, "abcd", [-0.3, -0.2, -0.38, -0.3]):
    tag(a, t, x)
h, l = handles(order)
fig.legend(h, l, ncol=9, loc="upper center", bbox_to_anchor=(0.53, 1.0), fontsize=6.0, columnspacing=0.8, handlelength=2.0, frameon=False)
fig.savefig("../figs/Fig3_mc.png", dpi=600); fig.savefig("../figs/Fig3_mc.pdf"); plt.close(fig)

# =============================================================== Fig. 4
rs = json.load(open("../results/real_summary.json"))
fits = pickle.load(open("../results/gapfits.pkl", "rb"))
fig = plt.figure(figsize=(7.16, 1.85))
ax = [fig.add_axes([0.065, 0.22, 0.2, 0.68]), fig.add_axes([0.33, 0.22, 0.17, 0.68]),
      fig.add_axes([0.57, 0.22, 0.17, 0.68]), fig.add_axes([0.815, 0.22, 0.175, 0.68])]
for g, col, mk in [("80", C["blue"], "o"), ("240", C["red"], "s")]:
    r = [q for q in fits if q["geom"] == g and q["c"] == 1.0][0]
    ax[0].loglog(r["f"][::2], np.abs(r["Z"][::2]), mk, ms=2.2, color=col, mfc="none", mew=0.5, label="%s $\\mu$m gap" % g)
    ax[0].loglog(r["f"], np.abs(R.zmodel(r["xfull"], r["f"])), "-", color=C["black"], lw=0.8)
ax[0].plot([], [], "-", color=C["black"], lw=0.8, label="model (9)")
ax[0].set_xlabel("Frequency (Hz)"); ax[0].set_ylabel("$|Z|$ ($\\Omega$), 1 mM NaCl"); ax[0].legend(fontsize=5.6, loc="lower left", frameon=False)
DS = {"log-uniform": dict(c="#6E6E6E", m="o", f="white", ls="--", lab="log-uniform (= heuristic)"),
      "nominal": dict(c=C["blue"], m="D", f="white", ls="-.", lab="nominal (1 mM)"),
      "robust": dict(c=C["red"], m="*", f=C["red"], ls="-", lab="robust")}
for dn, s in DS.items():
    ax[1].plot([4, 5, 6], [1000 * rs[f"{dn}_{m}"]["wc"] for m in [4, 5, 6]], ls=s["ls"], color=s["c"], marker=s["m"],
               mfc=s["f"], mec=s["c"], ms=6.5 if s["m"] == "*" else 4, lw=1.1 if dn != "robust" else 1.6, label=s["lab"])
ax[1].set_xticks([4, 5, 6]); ax[1].set_xlabel("Number of bins $m$"); ax[1].set_ylabel("RMS dev., worst conc.")
ax[1].legend(fontsize=5.4, loc="center right", bbox_to_anchor=(1.0, 0.62), frameon=True, framealpha=0.9, edgecolor="none")
ops = [("single", "single", C["red"], ""), ("matched-ref", "matched ref.", "white", "////"), ("mismatched-ref", "mismatched ref.", "white", "xxxx")]
for i, (k, lab, fc, ht) in enumerate(ops):
    vals = [1000 * rs[f"op_{k}_{m}"] for m in [4, 5, 6]]
    ax[2].bar(np.arange(3) + (i - 1) * 0.26, vals, width=0.26, color=fc, edgecolor=C["red"] if k == "single" else C["black"], hatch=ht, lw=0.6, label=lab, zorder=2)
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["4", "5", "6"]); ax[2].set_xlabel("Number of bins $m$")
ax[2].set_ylabel("RMS dev. of $\\ln G$"); ax[2].set_ylim(0, 80); ax[2].legend(fontsize=5.4, loc="upper left", frameon=False)
ax[2].grid(axis="x", visible=False)
pv = 1000 * np.array(rs["pred_gap"]); em = 1000 * np.array(rs["emp_gap"])
co = np.polyfit(pv, em, 1); xx = np.linspace(pv.min() * 0.97, pv.max() * 1.02, 20)
ax[3].plot(xx, np.polyval(co, xx), color=C["gray"], ls="--", lw=0.8, zorder=1)
ax[3].plot(pv, em, "o", color=C["blue"], ms=4.5, mfc="white", mew=1.0, zorder=3)
off = {"80": (-5, -8), "120": (5, -3), "160": (5, -5), "200": (3, 4), "240": (-3, -10)}
for g, x, y in zip(["80", "120", "160", "200", "240"], pv, em):
    ax[3].annotate(g, (x, y), textcoords="offset points", xytext=off[g], fontsize=5.6, ha="left" if off[g][0] >= 0 else "right")
ax[3].text(0.04, 0.95, "Spearman $\\rho$ = %.1f\n($p$ = %.2f, 5 gaps)" % (rs["spearman"], rs["spearman_p"]), transform=ax[3].transAxes, fontsize=5.4, va="top")
ax[3].set_xlabel("Predicted worst-case CRLB"); ax[3].set_ylabel("Observed RMS dev.")
ax[3].set_ylim(2.5, 17)
for a, t in zip(ax, "abcd"):
    tag(a, t, -0.3)
for a in ax[1:]:
    a.text(0.0, 1.01, "$\\times10^{-3}$", transform=a.transAxes, fontsize=5.8, va="bottom", ha="left")
fig.savefig("../figs/Fig4_real.png", dpi=600); fig.savefig("../figs/Fig4_real.pdf"); plt.close(fig)
print("ok")
