import json, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse, Arc
import figstyle as S
S.setup(); C = S.C
BLUE, RED, GRAY = C["blue"], C["red"], "#9A9A9A"
SUB = "#E9E4D8"; LIQ = "#C9D9EA"

des = {(r["name"], r["m"]): r for r in json.load(open("../results/designs.json"))}
N, w, s, Dout = des[("Ours", 6)]["d"]; w *= 1e3; s *= 1e3; Dout *= 1e3   # mm
freqs = des[("Ours", 6)]["freqs"]


def spiral_poly(N, w, s, Dout, phase, cx=0, cy=0):
    p = 2 * (w + s); r0 = Dout / 2 - w / 2
    phi = np.linspace(0, 2 * np.pi * N, 1200); r = r0 - p * phi / (2 * np.pi)
    a = phi + np.pi * phase
    o = np.c_[cx + (r + w / 2) * np.cos(a), cy + (r + w / 2) * np.sin(a)]
    i = np.c_[cx + (r - w / 2) * np.cos(a), cy + (r - w / 2) * np.sin(a)]
    return np.vstack([o, i[::-1]]), (cx + r0 * np.cos(np.pi * phase), cy + r0 * np.sin(np.pi * phase))


fig = plt.figure(figsize=(7.16, 3.0))

# ------------------------------------------------------------ (a) chip top view
ax = fig.add_axes([0.0, 0.40, 0.40, 0.56]); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(FancyBboxPatch((-0.3, -3.35), 9.2, 6.25, boxstyle="round,pad=0.05,rounding_size=0.3", fc=SUB, ec=C["black"], lw=0.7))
for k, (cx, lab, fill) in enumerate([(2.0, "Sample spiral", LIQ), (6.6, "Reference spiral", "#D8E8D8")]):
    if k == 0:
        ax.add_patch(Ellipse((cx, 0.35), Dout + 0.9, Dout + 0.75, fc=LIQ, ec=BLUE, lw=0.5, alpha=0.75, zorder=1))
        ax.text(cx, 0.35 + Dout / 2 + 0.5, "sample, $\\sigma^*(\\omega;\\theta)$", ha="center", fontsize=6, color=BLUE)
    else:
        ax.add_patch(Rectangle((cx - Dout / 2 - 0.45, 0.35 - Dout / 2 - 0.4), Dout + 0.9, Dout + 0.8, fc="#DCE6D5", ec=C["green"], lw=0.5, ls="--", zorder=1))
        ax.text(cx, 0.35 + Dout / 2 + 0.5, "sealed buffer, $\\sigma_r$ known", ha="center", fontsize=6, color=C["green"])
    for ph, col in [(0, BLUE), (1, RED)]:
        poly, end = spiral_poly(N, w, s, Dout, ph, cx, 0.35)
        ax.add_patch(Polygon(poly, closed=True, fc=col, ec="none", zorder=3))
        px = cx + 0.75 - 1.5 * ph
        side = 1 if ph == 0 else -1
        xo = cx + side * (Dout / 2 + 0.3)
        ax.plot([end[0], xo, xo, px, px], [end[1], end[1], -2.05, -2.05, -2.3], color=col, lw=1.3, zorder=2, solid_capstyle="butt")
        ax.add_patch(Rectangle((px - 0.32, -2.85), 0.64, 0.6, fc="#B08D3C", ec=C["black"], lw=0.5, zorder=3))
    ax.text(cx, -3.2, lab, ha="center", fontsize=6.5)
ax.plot([0.9, 3.1], [0.35, 0.35], color=C["black"], lw=0.6, ls=(0, (3, 2)), zorder=4)
ax.text(0.75, 0.35, "A", fontsize=6.5, ha="right", va="center"); ax.text(3.25, 0.35, "A$'$", fontsize=6.5, va="center")
ax.annotate("", xy=(2 - Dout / 2, -1.45), xytext=(2 + Dout / 2, -1.45), arrowprops=dict(arrowstyle="<->", lw=0.6))
ax.text(2, -1.75, "$D_{out}$=%.2f mm" % Dout, ha="center", fontsize=5.8)
ax.set_xlim(-0.5, 9.1); ax.set_ylim(-3.45, 3.05)
ax.text(-0.75, 3.3, "(a)", fontsize=8, va="top")

# ------------------------------------------------------------ (b) cross-section with field lines
ax = fig.add_axes([0.42, 0.42, 0.29, 0.54]); ax.axis("off")
pitch = w + s
xs = np.arange(-3, 4) * pitch
ax.add_patch(Rectangle((-3.4 * pitch, -0.55), 6.8 * pitch, 0.55, fc=SUB, ec="none"))
ax.add_patch(Rectangle((-3.4 * pitch, 0), 6.8 * pitch, 0.95, fc=LIQ, ec="none"))
for i, x in enumerate(xs):
    col = BLUE if i % 2 == 0 else RED
    ax.add_patch(Rectangle((x - w / 2, 0), w, 0.045, fc=col, ec="none", zorder=4))
    if i < len(xs) - 1:
        xc = x + pitch / 2
        for rr in np.linspace(s / 2 + 0.02, pitch / 2 + w / 2 - 0.02, 4):
            ax.add_patch(Arc((xc, 0.045), 2 * rr, 2 * rr * 0.95, theta1=0, theta2=180, lw=0.45, color=C["black"], alpha=0.55, zorder=3))
        ax.add_patch(FancyArrowPatch((xc - 0.004, 0.045 + (s / 2 + 0.08) * 0.95), (xc + 0.004, 0.045 + (s / 2 + 0.08) * 0.95),
                                     arrowstyle="-|>", mutation_scale=4, lw=0.4, color=C["black"], zorder=3))
x0 = xs[3]
ax.annotate("", xy=(x0 - w / 2, -0.12), xytext=(x0 + w / 2, -0.12), arrowprops=dict(arrowstyle="<->", lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=4))
ax.text(x0, -0.24, "$w$", ha="center", fontsize=6.5)
ax.annotate("", xy=(x0 + w / 2, -0.12), xytext=(x0 + w / 2 + s, -0.12), arrowprops=dict(arrowstyle="<->", lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=4))
ax.text(x0 + w / 2 + s / 2 + 0.03, -0.27, "$s$", ha="center", fontsize=6.5)
ax.annotate("", xy=(3.3 * pitch, 0.0), xytext=(3.3 * pitch, 0.95), arrowprops=dict(arrowstyle="<->", lw=0.6))
ax.text(3.22 * pitch, 0.5, "$H$", ha="right", fontsize=6.5)
ax.text(-3.3 * pitch, 0.82, "sample $\\sigma^*(\\omega)$", fontsize=6, color=BLUE)
ax.text(-3.3 * pitch, -0.42, "substrate $\\varepsilon_{sub}$", fontsize=6)
ax.text(0.45 * pitch, 1.03, "Section A–A$'$: periodic coplanar strips $\\Rightarrow K(\\mathbf{d})$", ha="center", fontsize=6.3)
ax.set_xlim(-3.45 * pitch, 3.45 * pitch); ax.set_ylim(-0.6, 1.15)
ax.text(-3.75 * pitch, 1.17, "(b)", fontsize=8, va="top")

# ------------------------------------------------------------ (c) equivalent circuit
ax = fig.add_axes([0.73, 0.40, 0.27, 0.58]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
lk = dict(color=C["black"], lw=0.8)
def res(x0, y, x1, lab, vertical=False, col=C["black"]):
    n = 6
    if not vertical:
        xx = np.linspace(x0 + 0.25, x1 - 0.25, 2 * n + 1); yy = y + np.r_[0, np.tile([0.18, -0.18], n)][:len(xx)]
        ax.plot([x0, x0 + 0.25], [y, y], **lk); ax.plot([x1 - 0.25, x1], [y, y], **lk); ax.plot(xx, yy, color=col, lw=0.8)
        ax.text((x0 + x1) / 2, y + 0.32, lab, ha="center", fontsize=6, color=col)
    else:
        yy = np.linspace(x0 + 0.25, x1 - 0.25, 2 * n + 1); xx = y + np.r_[0, np.tile([0.18, -0.18], n)][:len(yy)]
        ax.plot([y, y], [x0, x0 + 0.25], **lk); ax.plot([y, y], [x1 - 0.25, x1], **lk); ax.plot(xx, yy, color=col, lw=0.8)
        ax.text(y + 0.3, (x0 + x1) / 2, lab, ha="left", va="center", fontsize=6, color=col)
def ind(x0, y, x1, lab):
    ax.plot([x0, x0 + 0.2], [y, y], **lk); ax.plot([x1 - 0.2, x1], [y, y], **lk)
    n = 4; L = (x1 - x0 - 0.4) / n
    for k in range(n):
        ax.add_patch(Arc((x0 + 0.2 + L * (k + 0.5), y), L, 0.5, theta1=0, theta2=180, lw=0.8, color=C["black"]))
    ax.text((x0 + x1) / 2, y + 0.38, lab, ha="center", fontsize=6)
def cap_v(x, y0, y1, lab, col=C["black"]):
    ym = (y0 + y1) / 2
    ax.plot([x, x], [y0, ym - 0.1], **lk); ax.plot([x, x], [ym + 0.1, y1], **lk)
    ax.plot([x - 0.35, x + 0.35], [ym - 0.1, ym - 0.1], color=col, lw=1.0); ax.plot([x - 0.35, x + 0.35], [ym + 0.1, ym + 0.1], color=col, lw=1.0)
    ax.text(x + 0.45, ym, lab, va="center", fontsize=6, color=col)
def box_v(x, y0, y1, lab, col):
    ax.plot([x, x], [y0, y0 + 0.3], **lk); ax.plot([x, x], [y1 - 0.3, y1], **lk)
    ax.add_patch(Rectangle((x - 0.32, y0 + 0.3), 0.64, y1 - y0 - 0.6, fc="white", ec=col, lw=0.9))
    ax.text(x + 0.45, (y0 + y1) / 2, lab, va="center", fontsize=6, color=col)
ytop, ybot = 5.6, 0.8
ax.add_patch(Circle((0.5, ytop), 0.12, fc="white", ec=C["black"], lw=0.8)); ax.add_patch(Circle((0.5, ybot), 0.12, fc="white", ec=C["black"], lw=0.8))
res(0.62, ytop, 2.3, "$R_\\ell$"); ind(2.3, ytop, 3.9, "$L_\\ell$")
ax.plot([3.9, 4.3], [ytop, ytop], **lk); ax.plot([0.62, 8.8], [ybot, ybot], **lk)
cap_v(4.3, ybot, ytop, "$C_{fx}$")
res(4.3, ytop, 6.0, "$R_{tr}$", col=C["brown"])
cap_v(6.6, ybot, ytop, "$C_{sub}$", col=C["brown"])
box_v(8.8, 3.25, ytop, "$Z_{ep}$", RED)
box_v(8.8, ybot, 3.25, "$K/\\sigma^*$", BLUE)
ax.plot([6.0, 8.8], [ytop, ytop], **lk)
ax.text(0.15, 3.2, "$Z_s$ or $Z_r$", rotation=90, va="center", fontsize=6)
ax.text(5.2, 0.15, "fixture / leads  |  device  |  interface + sample", ha="center", fontsize=5.4, color=C["gray"])
ax.text(0, 7.0, "(c)", fontsize=8, va="top")

# ------------------------------------------------------------ (d) co-design pipeline
ax = fig.add_axes([0.0, 0.0, 1.0, 0.36]); ax.axis("off"); ax.set_xlim(0, 20); ax.set_ylim(0, 4)
def blk(x, y, ww, hh, txt, col=C["black"], fc="white", fs=6.2):
    ax.add_patch(FancyBboxPatch((x, y), ww, hh, boxstyle="round,pad=0.04,rounding_size=0.15", fc=fc, ec=col, lw=0.8))
    ax.text(x + ww / 2, y + hh / 2, txt, ha="center", va="center", fontsize=fs, color=col)
def ar(x0, y0, x1, y1, col=C["black"], ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=col, linestyle=ls))
blk(0.05, 1.5, 2.55, 1.7, "Uncertainty\n$\\theta\\in\\Theta$ (box)\n$\\nu\\sim\\mathcal{N}(\\bar\\nu,\\Sigma_\\nu)$")
blk(3.05, 1.5, 3.05, 1.7, "Operator & readout\nsingle or differential\n(digital / analog)\nnoise $\\eta|Z|+\\eta_0$", fs=5.9)
blk(6.55, 1.5, 3.75, 1.7, "Marginalized FIM\n$\\mathbf{I}=\\mathbf{J}_\\theta^T(\\Sigma+\\mathbf{J}_\\nu\\Sigma_\\nu\\mathbf{J}_\\nu^T)^{-1}\\mathbf{J}_\\theta$\nProp. 1: diff. vs single", BLUE)
blk(10.85, 2.5, 4.35, 1.1, "Convex frequency relaxation (Prop. 2)\n$\\max_{\\mathbf{w}}\\ \\min_t\\ \\lambda_{\\min}(\\mathbf{S}_t(\\mathbf{w}))$", RED, fs=5.8)
blk(10.85, 1.2, 4.35, 1.0, "Geometry search: enumerate $N$,\nsimplex search on $(w, s, D_{out})$", RED, fs=5.8)
blk(10.85, 0.05, 4.35, 0.85, "Exchange: worst case $\\theta^\\dagger\\in\\Theta$", RED, fs=5.8)
ar(2.6, 2.35, 3.05, 2.35); ar(6.1, 2.35, 6.55, 2.35); ar(10.3, 2.6, 10.85, 3.0); ar(10.3, 2.1, 10.85, 1.7)
ar(13.0, 2.5, 13.0, 2.2); ar(13.0, 1.2, 13.0, 0.9)
ax.plot([15.2, 15.55, 15.55, 15.2], [0.45, 0.45, 3.05, 3.05], color=C["black"], lw=0.8, ls="--")
ar(15.56, 3.05, 15.2, 3.05, ls="--")
ax.text(15.72, 1.05, "add scenario", fontsize=5.3, va="center", rotation=90)
ar(15.55, 2.0, 16.35, 2.0)
ax.text(15.62, 2.12, "converged", fontsize=5.0, ha="left")
# output: selected frequencies on log axis
ox0, ox1, oy = 16.6, 19.85, 1.5
ax.text((ox0 + ox1) / 2, 3.45, "Output: $\\mathbf{d}^\\star$, $F^\\star$ ($m$=6)", ha="center", fontsize=6.5, color=RED)
ax.text((ox0 + ox1) / 2, 2.95, "$N$=%d, $w$=%d $\\mu$m, $s$=%d $\\mu$m" % (N, round(w * 1e3), round(s * 1e3)), ha="center", fontsize=6)
ax.plot([ox0, ox1], [oy, oy], color=C["black"], lw=0.8)
for dec in range(3, 8):
    xx = ox0 + (dec - 3) / 4 * (ox1 - ox0)
    ax.plot([xx, xx], [oy - 0.12, oy + 0.12], color=C["black"], lw=0.6)
    ax.text(xx, oy - 0.5, "$10^{%d}$" % dec, ha="center", fontsize=5.6)
for f in freqs:
    xx = ox0 + (np.log10(f) - 3) / 4 * (ox1 - ox0)
    ax.plot([xx], [oy + 0.35], "v", color=RED, ms=4)
ax.text((ox0 + ox1) / 2, oy - 1.05, "frequency (Hz)", ha="center", fontsize=5.8)
ax.text(0.1, 3.95, "(d)", fontsize=8, va="top")

fig.savefig("../figs/Fig1_concept.png", dpi=600); fig.savefig("../figs/Fig1_concept.pdf")
print("ok")
