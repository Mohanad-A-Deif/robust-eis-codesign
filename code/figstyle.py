import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = dict(black="#000000", red="#8B0000", blue="#00205B", green="#0B5D1E", gray="#555555",
         purple="#3B0A57", brown="#5B3A1A", teal="#004D4D")
METHOD_STYLE = {
    "B1": dict(color=C["gray"], marker="o", ls="--", mfc="white"),
    "B2": dict(color=C["gray"], marker="s", ls=":", mfc="white"),
    "B3": dict(color=C["brown"], marker="^", ls="--", mfc="white"),
    "B4": dict(color=C["brown"], marker="v", ls=":", mfc="white"),
    "B5": dict(color=C["blue"], marker="D", ls="-.", mfc="white"),
    "B6": dict(color=C["green"], marker="P", ls="-.", mfc="white"),
    "B7": dict(color=C["purple"], marker="X", ls="--", mfc=C["purple"]),
    "Ours": dict(color=C["red"], marker="*", ls="-", mfc=C["red"], ms=9),
    "Ours-dig": dict(color=C["red"], marker="*", ls=":", mfc="white", ms=9),
    "B3-dig": dict(color=C["brown"], marker="^", ls="-", mfc=C["brown"]),
}


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7, "lines.linewidth": 1.1,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight", "figure.dpi": 150,
    })
