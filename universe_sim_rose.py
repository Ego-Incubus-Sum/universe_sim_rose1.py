"""
universe_sim_rose.py -- mirrored (flower / rose) views of the universe_sim.py diagram.

View 1: the original four-panel figure tiled with mirror copies (4-fold mirror symmetry
        around the centre, like looking at a flower's core).
View 2: a polar rose: each module's curve becomes one petal; every petal is mirrored about
        its own axis. Module B (tear detector) is the highlighted petal.

The symmetry is imposed by construction (a presentation choice). It is not a property of
the physics.
"""
import io, contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ---- reuse the simulation results without re-printing the logs
with contextlib.redirect_stdout(io.StringIO()):
    import universe_sim as S

OUT = "/mnt/user-data/outputs/"

# ===================================================== View 1: mirrored tiling
img = mpimg.imread(OUT + "universe_sim.png")
top = np.concatenate([img, img[:, ::-1]], axis=1)             # original | left-right mirror
full = np.concatenate([top, top[::-1]], axis=0)               # top / up-down mirror
fig = plt.figure(figsize=(14, 9.7))
plt.imshow(full)
plt.axis("off")
plt.title("Universe simulation diagram, 4-fold mirror view (centre = flower core)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT + "universe_sim_mirrored.png", dpi=110)
plt.close(fig)

# ===================================================== View 2: polar rose
def norm(y):
    y = np.asarray(y, float)
    y = np.where(np.isfinite(y), y, np.nan)
    lo, hi = np.nanmin(y), np.nanmax(y)
    return (y - lo) / (hi - lo + 1e-30)

def resample(x, y, n=200):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    xs = np.linspace(x[ok].min(), x[ok].max(), n)
    return np.linspace(0, 1, n), np.interp(xs, x[ok], y[ok])

petals = {
    "A1 collapse test":   [resample(S.t_obs, S.a_obs), resample(S.t_hyp, S.a_hyp)],
    "A2 infall / Hubble": [resample(S.d, S.ratio)],
    "B tear detector":    [resample(S.good.t, S.good.y[0]),
                           resample(*S.euler(1.0)[:2])],
    "C lineage":          [resample(np.arange(len(S.mean_f)), S.mean_f)],
}
centres = {"A1 collapse test": 0, "A2 infall / Hubble": 90,
           "B tear detector": 180, "C lineage": 270}
colors = {"A1 collapse test": "#d95f8d", "A2 infall / Hubble": "#e8a0bf",
          "B tear detector": "#c1121f", "C lineage": "#f4c2d7"}

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="polar")
r0, half = 0.18, np.deg2rad(42)

for name, curves in petals.items():
    th0 = np.deg2rad(centres[name])
    for k, (x, y) in enumerate(curves):
        yn = norm(y)
        r = r0 + (1 - r0) * np.nan_to_num(yn)
        th = x * half
        # mirror about the petal's own axis
        th_full = np.concatenate([th0 - th[::-1], th0 + th])
        r_full = np.concatenate([r[::-1], r])
        main = (name == "B tear detector" and k == 0) or (name != "B tear detector" and k == 0)
        ax.fill(th_full, r_full, color=colors[name], alpha=0.55 if k == 0 else 0.0)
        ax.plot(th_full, r_full, color="#c1121f" if name == "B tear detector" and k == 1
                else "#5a0f2a", lw=2.2 if name == "B tear detector" else 1.3,
                ls="--" if (name == "B tear detector" and k == 1) or
                           (name == "A1 collapse test" and k == 1) else "-")
    ax.text(th0, 1.22, name, ha="center", va="center", fontsize=10,
            fontweight="bold" if name == "B tear detector" else "normal",
            color="#c1121f" if name == "B tear detector" else "#333")

# flower core
core = np.linspace(0, 2 * np.pi, 200)
ax.fill(core, np.full_like(core, r0), color="#f9d949", zorder=3)
ax.set_ylim(0, 1.3)
ax.set_yticks([]); ax.set_xticks([])
ax.spines["polar"].set_visible(False)
ax.set_title("Universe simulation as a rose: one mirrored petal per module\n"
             "B (red): stable bounce (solid) vs unstable run caught by the tear detector (dashed)",
             fontsize=11, pad=20)
plt.tight_layout()
plt.savefig(OUT + "universe_sim_rose.png", dpi=140)
print("saved universe_sim_mirrored.png and universe_sim_rose.png")
