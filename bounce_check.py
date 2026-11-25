"""
Numerical checks for the "universe inside a black hole" argument.

Part 1: Is r_s(M_Hubble) = c/H for a flat universe?
Part 2: Integrate the flat LambdaCDM Friedmann equation and compare to the analytic solution.
Part 3: Integrate a loop-quantum-cosmology (LQC) bounce:
            H^2 = (8 pi G / 3) rho (1 - rho / rho_crit)
        in Planck units, for pressureless matter, through the bounce.

Part 3 is a toy model: it shows the math of a bounce, not that one happened.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------- Part 1
G = 6.67430e-11          # m^3 kg^-1 s^-2
c = 2.99792458e8         # m/s
Mpc = 3.0856775814913673e22
H0 = 67.4e3 / Mpc        # s^-1

rho_crit = 3 * H0**2 / (8 * np.pi * G)          # critical density, kg/m^3
R_H = c / H0                                     # Hubble radius, m
M_H = rho_crit * (4 / 3) * np.pi * R_H**3        # mass inside Hubble sphere, kg
r_s = 2 * G * M_H / c**2                         # Schwarzschild radius of that mass

print("=== Part 1: Hubble-radius check ===")
print(f"Critical density      : {rho_crit:.3e} kg/m^3")
print(f"Hubble radius c/H0    : {R_H:.6e} m")
print(f"Mass in Hubble sphere : {M_H:.6e} kg")
print(f"Schwarzschild radius  : {r_s:.6e} m")
print(f"r_s / (c/H0)          : {r_s / R_H:.12f}  (should be 1)")
print("Note: this follows algebraically from flatness (rho = rho_crit).")
print("It does NOT distinguish 'inside a black hole' from 'ordinary flat universe'.\n")

# ---------------------------------------------------------------- Part 2
Om, OL = 0.315, 0.685   # flat, radiation neglected

def friedmann(t, y):
    a = y[0]
    H = H0 * np.sqrt(Om / a**3 + OL)
    return [a * H]

t_end = 3.0 / H0
sol = solve_ivp(friedmann, [0, t_end], [1e-6], rtol=1e-10, atol=1e-14,
                dense_output=True, max_step=t_end / 4000)

# Analytic flat LCDM: a(t) = (Om/OL)^(1/3) * sinh(1.5 * sqrt(OL) * H0 * t)^(2/3)
def a_analytic(t):
    return (Om / OL) ** (1 / 3) * np.sinh(1.5 * np.sqrt(OL) * H0 * t) ** (2 / 3)

ts = np.linspace(0.05 / H0, t_end, 400)
a_num = sol.sol(ts)[0]
a_ana = a_analytic(ts)
rel_err = np.max(np.abs(a_num - a_ana) / a_ana)

# Age of the universe where a = 1
t_now = (2 / (3 * np.sqrt(OL) * H0)) * np.arcsinh(np.sqrt(OL / Om))
Gyr = 3.15576e16
print("=== Part 2: Flat LambdaCDM ===")
print(f"Max relative error numeric vs analytic a(t): {rel_err:.2e}")
print(f"Age at a=1 (analytic): {t_now / Gyr:.2f} Gyr\n")

# ---------------------------------------------------------------- Part 3
# Planck units: G = hbar = c = 1.  LQC critical density ~ 0.41 rho_Planck.
G_p = 1.0
rho_c = 0.41
rho0 = 1e-6 * rho_c     # dust: rho = rho0 / a^3 (arbitrary normalization, a=1 at rho=rho0)

def rho_of(a):
    return rho0 / a**3

def lqc(t, y):
    """y = [a, H]. Dust: P = 0.
    da/dt = a H
    dH/dt = -4 pi G (rho + P) (1 - 2 rho / rho_c)   (LQC Raychaudhuri equation)
    """
    a, H = y
    rho = rho_of(a)
    dH = -4 * np.pi * G_p * rho * (1 - 2 * rho / rho_c)
    return [a * H, dH]

# Start in the contracting phase at low density (a large, H < 0).
a_start = (rho0 / (1e-4 * rho_c)) ** (1 / 3)
rho_start = rho_of(a_start)
H_start = -np.sqrt((8 * np.pi * G_p / 3) * rho_start * (1 - rho_start / rho_c))

# Time scale: ~ 1/sqrt(G rho_c) in Planck units; integrate long enough to cross the bounce
t_span = (0, 2.0e4)
sol_b = solve_ivp(lqc, t_span, [a_start, H_start], rtol=1e-10, atol=1e-14,
                  dense_output=True, max_step=1.0)
tb = sol_b.t
ab, Hb = sol_b.y
rho_b = rho_of(ab)

i_min = np.argmin(ab)
print("=== Part 3: LQC bounce (Planck units, dust) ===")
print(f"Initial a          : {a_start:.3f}, H = {H_start:.3e} (contracting)")
print(f"Minimum scale factor: {ab[i_min]:.4f} at t = {tb[i_min]:.1f}")
print(f"Max density reached : {rho_b.max():.6f}  vs rho_crit = {rho_c}")
print(f"H at bounce         : {Hb[i_min]:.3e}  (should be ~0)")
print(f"Final H sign        : {'expanding' if Hb[-1] > 0 else 'contracting'}")
# Constraint check: H^2 should equal (8 pi G/3) rho (1 - rho/rho_c) all along the path
constraint = Hb**2 - (8 * np.pi * G_p / 3) * rho_b * (1 - rho_b / rho_c)
print(f"Max |constraint violation| along run: {np.max(np.abs(constraint)):.2e}")
print("A bounce occurs only because the model has the (1 - rho/rho_c) term built in.")
print("Classical GR (without it) would run to a = 0 with rho -> infinity.\n")

# ---------------------------------------------------------------- Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel 1: r_s vs c/H across a range of H (flat universe, rho = rho_crit)
Hs = np.logspace(-20, -17, 50)
rs_arr = 2 * G * (3 * Hs**2 / (8 * np.pi * G)) * (4 / 3) * np.pi * (c / Hs) ** 3 / c**2
axes[0].loglog(c / Hs, rs_arr, label="r_s of Hubble-sphere mass")
axes[0].loglog(c / Hs, c / Hs, "--", label="c/H")
axes[0].set_xlabel("Hubble radius c/H (m)")
axes[0].set_ylabel("radius (m)")
axes[0].set_title("Any flat universe satisfies r_s = c/H")
axes[0].legend()

# Panel 2: LCDM
axes[1].plot(ts * H0, a_num, label="numerical")
axes[1].plot(ts * H0, a_ana, "--", label="analytic")
axes[1].set_xlabel("H0 t")
axes[1].set_ylabel("a(t)")
axes[1].set_title("Flat ΛCDM: numerical vs analytic")
axes[1].legend()

# Panel 3: bounce
axes[2].plot(tb, ab, color="tab:red")
axes[2].axvline(tb[i_min], color="gray", ls=":")
axes[2].set_xlabel("t (Planck units)")
axes[2].set_ylabel("a(t)")
axes[2].set_title("LQC toy bounce: a(t) reaches a minimum, then expands")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/bounce_check.png", dpi=140)
print("Saved plot: /mnt/user-data/outputs/bounce_check.png")
