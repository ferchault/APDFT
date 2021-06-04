#!/usr/bin/env python

#%%
import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt
import glob

# %%
def get_nuclear(lval: float) -> float:
    """ Works only for N2 and only for this geometry."""
    d_au = 1.8897259886
    return (7 - lval) * (7 + lval) / d_au


def get_energy(lval: str, label: str) -> float:
    with open(f"RUN_{lval}/{label}.log") as fh:
        lines = fh.readlines()
    relevant = [_ for _ in lines if "Total energy:" in _][0]
    energy = float(relevant.strip().split()[-1])
    return energy


def get_derivative(lval: str) -> float:
    deltaNN = get_nuclear(1) - get_nuclear(0)
    return get_energy(lval, "target") - get_energy(lval, "reference") - deltaNN


# %%
lvals = [_[4:] for _ in sorted(glob.glob("RUN_*"), key=lambda _: float(_[4:]))]
xs = [float(_) for _ in lvals]
ys = [get_energy(_, "scf") - get_nuclear(float(_)) for _ in lvals]
# %%
plt.plot(xs, ys, lw=4, label="SCF")
plt.xlabel("Lambda for N2 -> CO")
plt.ylabel("Electronic energy [Ha]")

label = "HF gradients"
for lval in lvals:
    deriv = get_derivative(lval)
    center = get_energy(lval, "scf") - get_nuclear(float(lval))
    delta = 0.4
    plt.plot(
        (float(lval) - delta, float(lval) + delta),
        (center - delta * deriv, center + delta * deriv),
        color="red",
        label=label,
        alpha=0.4,
    )
    label = None
plt.show()

# comparison of orders
plt.plot(xs, ys, label="SCF")
plt.xlabel("Lambda for N2 -> CO")
plt.ylabel("Electronic energy [Ha]")
xss = np.linspace(-1, 1, 100)
d0 = get_energy("0.0", "scf") - get_nuclear(0)
d1 = get_derivative("0.0")
d2 = (get_derivative("0.01") - get_derivative("-0.01")) / 0.02
d3 = (get_derivative("0.01") - 2 * get_derivative("0.0") + get_derivative("-0.01")) / (
    0.01 ** 2
)
d4 = (
    0.5 * get_derivative("0.02")
    - 1 * get_derivative("0.01")
    + get_derivative("-0.01")
    - 0.5 * get_derivative("-0.02")
) / (0.01 ** 3)
plt.plot(xss, d0 + xss * 0, label="APDFT0")
plt.plot(xss, d0 + d1 * xss, label="APDFT1")
plt.plot(xss, d0 + d1 * xss + d2 / 2 * xss ** 2, label="APDFT2")
plt.legend()
plt.show()

# delta between orders

interpolant = sci.interp1d(xs, ys, "cubic")
plt.plot(xs, ys - interpolant(xs), label="SCF")
plt.xlabel("Lambda for N2 -> CO")
plt.ylabel("Energy error [Ha]")
xss = np.linspace(-1, 1, 100)
plt.plot(xss, -interpolant(xss) + d0 + xss * 0, label="APDFT0")
plt.plot(xss, -interpolant(xss) + d0 + d1 * xss, label="APDFT1")
plt.plot(xss, -interpolant(xss) + d0 + d1 * xss + d2 / 2 * xss ** 2, label="APDFT2")
plt.plot(
    xss,
    -interpolant(xss) + d0 + d1 * xss + d2 / 2 * xss ** 2 + d3 / 6 * xss ** 3,
    label="APDFT3",
)
plt.plot(
    xss,
    -interpolant(xss)
    + d0
    + d1 * xss
    + d2 / 2 * xss ** 2
    + d3 / 6 * xss ** 3
    + d4 / 24 * xss ** 4,
    label="APDFT4",
)
plt.ylim(-0.01, 0.05)
plt.legend()
