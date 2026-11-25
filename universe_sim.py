"""
universe_sim.py  --  modular simulation for the Adam / Eve / Garden framework.

Modules
  A  Adam's link  : does the environment collapse inward?  (recollapse test + local infall test)
  B  Eve's engine : LQC-style bounce with a "tear detector" (numerical-stability monitor)
  C  Garden       : lineage tree along one axis, up = ancestors, down = descendants,
                    with Smolin-style mutation of a 'fecundity' parameter

Every module prints a status line: DERIVED (follows from equations), INPUT (you set it),
or CHECK (a test the claim can pass or fail). Literature numbers marked (approx) should be
verified before being relied on.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------------------------------------------ constants
G = 6.67430e-11
c = 2.99792458e8
Mpc = 3.0856775814913673e22
Msun = 1.98847e30
H0 = 67.4e3 / Mpc
H0_kms_Mpc = 67.4

# =====================================================================
# MODULE A1: does the whole environment collapse inward?
#   Flat, dust + Lambda, units H0 = 1:   a'' = -0.5*Om*a^-2 + OL*a
#   Observed:      Om = 0.315, OL = 0.685
#   Hypothetical:  OL < 0 (flat requires Om = 1 - OL)  -> recollapse
# =====================================================================
def run_cosmos(Om, OL, t_max=6.0):
    def f(t, y):
        a, adot = y
        return [adot, -0.5 * Om / a**2 + OL * a]
    a0 = 1e-3
    adot0 = np.sqrt(Om / a0 + OL * a0**2)          # H^2 a^2 = Om/a + OL a^2
    ev = lambda t, y: y[0] - 1e-3
    ev.terminal = True
    ev.direction = -1
    s = solve_ivp(f, [0, t_max], [a0, adot0], events=ev, rtol=1e-10,
                  atol=1e-13, max_step=0.01)
    return s.t, s.y[0]

t_obs, a_obs = run_cosmos(0.315, 0.685)
OL_h = -0.5
t_hyp, a_hyp = run_cosmos(1 - OL_h, OL_h)
a_max_pred = (-(1 - OL_h) / OL_h) ** (1 / 3)

print("=== MODULE A1: global collapse test ===")
print(f"[CHECK] Observed LCDM: a(t) still growing at end of run? "
      f"{a_obs[-1] > a_obs[-50]}  (a_end = {a_obs[-1]:.2f})")
print(f"[CHECK] Observed LCDM has a turning point (H=0)? "
      f"{'no' if np.all(0.315 / a_obs**3 + 0.685 > 0) else 'yes'}")
print(f"[INPUT] Hypothetical Lambda = {OL_h}: a_max predicted {a_max_pred:.3f}, "
      f"simulated {a_hyp.max():.3f}, recollapses: {a_hyp[-1] < a_hyp.max()}")
print("-> Inward collapse of the whole environment needs Lambda < 0 (or much more matter).")
print("   Measured Lambda is positive; the observed universe is NOT collapsing as a whole.\n")

# =====================================================================
# MODULE A2: are all clusters moving toward one point?
#   Compare peculiar (infall) velocity to the Hubble flow at the same distance.
# =====================================================================
v_pec = 620.0                       # km/s, Local Group motion in CMB frame (approx, ~627)
d = np.linspace(10, 300, 200)       # Mpc
v_hubble = H0_kms_Mpc * d
ratio = v_pec / v_hubble

# Compactness of the largest known attractor-scale concentration (approx literature values)
M_att = 1e17 * Msun                 # Laniakea-scale mass (approx)
R_att = 80 * Mpc                    # approx radius
r_s_att = 2 * G * M_att / c**2
compact = r_s_att / R_att

print("=== MODULE A2: 'everything moves toward one point' test ===")
print(f"[CHECK] Peculiar speed / Hubble speed at 50 Mpc : {v_pec/(H0_kms_Mpc*50):.2f}")
print(f"[CHECK]                              at 100 Mpc: {v_pec/(H0_kms_Mpc*100):.2f}")
print(f"[CHECK]                              at 300 Mpc: {v_pec/(H0_kms_Mpc*300):.3f}")
print(f"[CHECK] Attractor-scale mass {M_att/Msun:.0e} Msun, R = 80 Mpc: "
      f"r_s = {r_s_att/Mpc*1e3:.1f} kpc, compactness 2GM/(Rc^2) = {compact:.1e}")
print("-> Real: there are large-scale flows toward mass concentrations (Great Attractor /")
print("   Shapley region), and also flows away (Dipole Repeller). They are small corrections")
print("   on the overall expansion, and compactness is ~1e-4, far from a black hole (needs ~1).\n")

# =====================================================================
# MODULE B: Eve's engine: bounce + tear detector
#   H^2 = (8 pi/3) rho (1 - rho/rho_c)   (Planck units, dust)
#   Detector flags a 'critical error' if: NaN, a <= 0, or constraint violation > tol.
# =====================================================================
rho_c = 0.41
rho0 = 1e-6 * rho_c
rho_of = lambda a: rho0 / a**3

def rhs(t, y):
    a, H = y
    rho = rho_of(a)
    return [a * H, -4 * np.pi * rho * (1 - 2 * rho / rho_c)]

a_s = (rho0 / (1e-4 * rho_c)) ** (1 / 3)
H_s = -np.sqrt((8 * np.pi / 3) * rho_of(a_s) * (1 - rho_of(a_s) / rho_c))

def constraint(a, H):
    rho = rho_of(a)
    return H**2 - (8 * np.pi / 3) * rho * (1 - rho / rho_c)

def detector(t, a, H, tol=1e-6):
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(H))):
        return "CRITICAL ERROR (numerical): non-finite values"
    if np.any(a <= 0):
        return "CRITICAL ERROR (numerical): scale factor reached zero or went negative"
    v = np.max(np.abs(constraint(a, H)))
    if v > tol:
        return f"CRITICAL ERROR (numerical): constraint violation {v:.1e} > tol {tol:.0e}"
    return f"OK: constraint violation {v:.1e}, a_min = {a.min():.4f}"

# good run: adaptive RK45
good = solve_ivp(rhs, [0, 2e4], [a_s, H_s], rtol=1e-10, atol=1e-14, max_step=1.0)
msg_good = detector(good.t, good.y[0], good.y[1])

# bad run: explicit Euler with a step that is too large
def euler(dt, T=2e4):
    n = int(T / dt)
    a, H = a_s, H_s
    ts, as_, Hs = [0.0], [a], [H]
    for i in range(n):
        da, dH = rhs(0, [a, H])
        a, H = a + dt * da, H + dt * dH
        ts.append((i + 1) * dt); as_.append(a); Hs.append(H)
        if not np.isfinite(a) or a <= 0:
            break
    return np.array(ts), np.array(as_), np.array(Hs)

tb, ab, Hb = euler(dt=40.0)
msg_bad = detector(tb, ab, Hb)

print("=== MODULE B: bounce engine + tear detector ===")
print(f"[DERIVED] adaptive RK45 : {msg_good}")
print(f"[DERIVED] Euler, dt=40  : {msg_bad}")
print("-> A 'tear' in this simulation is numerical: constraint drift or a hitting zero.")
print("   Real torsion is a repulsive spin term in Einstein-Cartan gravity, not a tear.\n")

# =====================================================================
# MODULE C: Garden lineage along one axis
#   generation 0 = our universe; negative = ancestors (up), positive = descendants (down).
#   Smolin: daughter fecundity f' = f * (1 + sigma*xi); offspring ~ Poisson(f).
# =====================================================================
rng = np.random.default_rng(7)
K_UP, K_DOWN = 12, 12
CAP = 3000               # carrying capacity: finite, so 'infinite' is NOT simulated
sigma = 0.05

# upward chain: single ancestry line from the root (finite by construction)
f_line = [1.0]
for _ in range(K_UP):
    f_line.append(f_line[-1] * (1 + sigma * rng.standard_normal()))
f_ours = f_line[-1]

# downward: branching descendants of generation 0
pop = np.array([f_ours] * 20)
mean_f, sizes = [pop.mean()], [len(pop)]
for g in range(K_DOWN):
    kids = []
    for f in pop:
        n = rng.poisson(max(f, 0) * 1.3)
        kids.extend([f * (1 + sigma * rng.standard_normal()) for _ in range(n)])
    pop = np.array(kids) if kids else np.array([0.0])
    if len(pop) > CAP:
        pop = rng.choice(pop, CAP, replace=False)
    mean_f.append(pop.mean()); sizes.append(len(pop))

print("=== MODULE C: Garden lineage (one axis, up/down) ===")
print(f"[INPUT] upward chain length {K_UP} (finite root by construction); "
      f"downward {K_DOWN} generations, cap {CAP}")
print(f"[DERIVED] mean fecundity: gen 0 = {mean_f[0]:.3f}, gen {K_DOWN} = {mean_f[-1]:.3f}")
print("-> Single seed, so this is an illustration, not a statistical result: rerun with")
print("   many seeds before drawing conclusions about selection (Smolin's argument).")
print("   Infinity is never reached: both directions are capped, which is an INPUT.\n")

# =====================================================================
# Plots
# =====================================================================
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

ax[0, 0].plot(t_obs, a_obs, label="observed: Lambda = +0.685")
ax[0, 0].plot(t_hyp, a_hyp, label="hypothetical: Lambda = -0.5")
ax[0, 0].set_xlabel("H0 t"); ax[0, 0].set_ylabel("a(t)")
ax[0, 0].set_title("A1: only negative Lambda collapses the environment")
ax[0, 0].legend()

ax[0, 1].plot(d, ratio)
ax[0, 1].axhline(1, color="gray", ls=":")
ax[0, 1].set_xlabel("distance (Mpc)"); ax[0, 1].set_ylabel("v_pec / v_Hubble")
ax[0, 1].set_title("A2: infall is a small correction to expansion at large scales")

ax[1, 0].plot(good.t, good.y[0], label="RK45 (stable)")
t1, a1, H1 = euler(dt=1.0)      # finer fixed step: survives the contraction, then fails at the bounce
ax[1, 0].plot(t1, a1, "r--", label="Euler dt=1 (fails at bounce)")
ax[1, 0].plot(tb, ab, "r:", marker="x", label="Euler dt=40 (fails at step 2)")
ax[1, 0].axhline(0, color="gray", lw=0.8)
ax[1, 0].set_xlabel("t (Planck units)"); ax[1, 0].set_ylabel("a(t)")
ax[1, 0].set_title("B: tear detector catches the unstable run")
ax[1, 0].legend()

gens = np.arange(0, K_DOWN + 1)
ax[1, 1].plot(gens, mean_f, marker="o")
ax[1, 1].set_xlabel("descendant generation"); ax[1, 1].set_ylabel("mean fecundity")
ax[1, 1].set_title("C: lineage selection (finite, capped)")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/universe_sim.png", dpi=130)
print("Saved /mnt/user-data/outputs/universe_sim.png")
