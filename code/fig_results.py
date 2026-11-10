import json, numpy as np
import matplotlib.pyplot as plt
import figstyle as S
import model as M, design as D
S.setup(); C = S.C; ST = S.METHOD_STYLE

# ---------------------------------------------------------------- Fig. 2 FEM verification
fv = json.load(open("../results/fem_verif.json"))
r = np.array(fv["ratio"])
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.65), gridspec_kw=dict(wspace=0.45))
ax[0].plot(r, fv["analytic"], color=C["black"], lw=1.1, label="Conformal map")
for H, mk, col in [(5000, "o", C["red"]), (100, "s", C["blue"]), (50, "^", C["green"])]:
    ax[0].plot(r, fv["H%d" % H], mk, mfc="white", color=col, ms=3.5, label="FEM, $H$=%d $\\mu$m" % H if H < 5000 else "FEM, $H$=5 mm")
ax[0].set_xlabel("$w/(w+s)$"); ax[0].set_ylabel("$\\Phi=G'/\\sigma$")
ax[0].legend(loc="upper left", fontsize=5.6, handlelength=1.2)
for H, mk, col in [(5000, "o", C["red"]), (300, "D", C["black"]), (100, "s", C["blue"]), (50, "^", C["green"])]:
    ax[1].plot(r, 100 * (np.array(fv["H%d" % H]) / np.array(fv["analytic"]) - 1), mk + "-", mfc="white", color=col, ms=3, lw=0.7,
               label=("5 mm" if H == 5000 else "%d $\\mu$m" % H))
ax[1].set_xlabel("$w/(w+s)$"); ax[1].set_ylabel("FEM/analytic $-1$ (%)")
ax[1].legend(title="$H$ (pitch 100 $\\mu$m)", fontsize=5.6, title_fontsize=5.6, loc="center", bbox_to_anchor=(0.5, 0.55), ncol=2)
for a, t in zip(ax, "ab"):
    a.text(-0.3, 1.02, "(%s)" % t, transform=a.transAxes, fontsize=8)
fig.savefig("../figs/Fig2_fem.png"); fig.savefig("../figs/Fig2_fem.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3 budget curves + per-parameter precision
ev = json.load(open("../results/eval_fim.json"))
E = {(r["name"], r["m"]): r for r in ev}
names = ["B1", "B2", "B3", "B5", "B6", "B7", "Ours-dig", "Ours"]
labels = {"B1": "B1", "B2": "B2", "B3": "B3 (=B4)", "B5": "B5", "B6": "B6", "B7": "B7", "Ours": "Proposed (analog)", "Ours-dig": "Proposed (digital)"}
fig, ax = plt.subplots(1, 2, figsize=(7.16, 1.95), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.28))
ms = [4, 6, 8, 10]
for n in names:
    y = [np.mean(E[(n, m)]["ME"]) for m in ms]
    st = dict(ST[n]); st.setdefault("ms", 5)
    ax[0].semilogy(ms, y, label=labels[n], **st)
ax[0].set_xlabel("Frequency budget $m$"); ax[0].set_ylabel("Worst-case $\\lambda_{\\min}(\\mathbf{I})$ over $\\Theta$")
ax[0].set_xticks(ms); ax[0].legend(ncol=2, fontsize=5.8, loc="lower right", columnspacing=0.6)
bars = ["B1", "B3", "B5", "B6", "B7", "Ours-dig", "Ours"]
pn = ["$\\sigma_0$", "$\\kappa$", "$\\tau$", "$\\alpha$"]
wbar = 0.11
hatches = ["////", "....", "xxxx", "\\\\\\\\", "----", "oo", ""]
for i, n in enumerate(bars):
    v = np.mean(E[(n, 6)]["ratio"], 0)
    ax[1].bar(np.arange(4) + (i - 3) * wbar, v, width=wbar, color="white" if n != "Ours" else C["red"],
              edgecolor=ST[n]["color"], hatch=hatches[i], lw=0.7, label=labels[n])
ax[1].axhline(1, color=C["black"], ls="--", lw=0.8)
ax[1].set_yscale("log"); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(pn)
ax[1].set_ylabel("Worst-case CRLB / target ($m$=6)")
ax[1].legend(ncol=4, fontsize=5.6, loc="upper left", columnspacing=0.6)
ax[1].set_ylim(0.5, 200)
for a, t in zip(ax, "ab"):
    a.text(-0.13 if t == "a" else -0.09, 1.02, "(%s)" % t, transform=a.transAxes, fontsize=8)
fig.savefig("../figs/Fig3_fim.png"); fig.savefig("../figs/Fig3_fim.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 4 relaxed weights and selected frequencies
des = {(r["name"], r["m"]): r for r in json.load(open("../results/designs.json"))}
M.NOISE["readout"] = "analog"
ic = D.InfoCache("diff")
fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.25), sharex=True, gridspec_kw=dict(hspace=0.12, height_ratios=[1, 1.15]))
d = tuple(des[("Ours", 6)]["d"])
cols = {4: C["blue"], 6: C["red"], 8: C["green"], 10: C["black"]}
for m in [4, 6, 8, 10]:
    dm = tuple(des[("Ours", m)]["d"])
    Gs = [ic.G(dm, t) for t in D.corner_scenarios()]
    w = D.design_weights(Gs, ic.P, m, iters=600, rtol=0)
    ax[0].plot(D.FGRID, w + 0 * m, "-", color=cols[m], lw=0.8, label="$m$=%d" % m)
    sel = des[("Ours", m)]["idx"]
    ax[0].plot(D.FGRID[sel], np.full(len(sel), 1.08 + 0.07 * [4, 6, 8, 10].index(m)), "|", color=cols[m], ms=6, mew=1.2)
ax[0].set_xscale("log"); ax[0].set_ylabel("Relaxed weight $w_\\ell$"); ax[0].set_ylim(-0.03, 1.35)
ax[0].legend(ncol=2, fontsize=5.8, loc="center left", bbox_to_anchor=(0.12, 0.55), columnspacing=0.8, handlelength=1.2)
nu = M.nu_vector()
for th, ls, lab in [(np.array([M.THETA_BOX[0, 1], M.THETA_BOX[1, 0], M.THETA_BOX[2, 0], M.THETA_BOX[3, 1]]), "-", "worst corner"),
                    (M.THETA_C, "--", "center of $\\Theta$")]:
    G = ic.G(d, th)
    A = G.sum(0)
    for k, (pnm, col) in enumerate(zip(pn, [C["red"], C["blue"], C["green"], C["black"]])):
        ax[1].loglog(D.FGRID, np.sqrt(G[:, k, k]), ls, color=col, lw=0.8, label=(pnm if ls == "-" else None))
ax[1].set_ylabel("$\\sqrt{G_{\\ell,kk}}$"); ax[1].set_xlabel("Frequency (Hz)")
for f in des[("Ours", 6)]["freqs"]:
    ax[1].axvline(f, color=C["gray"], lw=0.6, ls=":")
ax[1].legend(ncol=4, fontsize=5.8, loc="lower left", columnspacing=0.8, handlelength=1.4)
ax[1].set_ylim(1e-2, None)
for a, t in zip(ax, "ab"):
    a.text(0.01, 0.97, "(%s)" % t, transform=a.transAxes, fontsize=8, va="top")
fig.savefig("../figs/Fig4_frequencies.png"); fig.savefig("../figs/Fig4_frequencies.pdf"); plt.close(fig)
print("ok")
