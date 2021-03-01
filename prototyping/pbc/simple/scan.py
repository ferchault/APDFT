#!/usr/bin/env python
#%%
import subprocess
import shlex
import json
import numpy as np
import matplotlib.pyplot as plt

#%%

UP = "Ni Cr Ti Mn Fe"
DN = "Ti Fe Ni Mn Cr"

UP2 = "Co Cr V Mn Fe"
DN2 = "V Fe Co Mn Cr"

UP3 = "F C B N O"
DN3 = "B O F N C"


def get_poscar(d, box, elements):
    return f"""generated
1
{d*5} 0 0
0 {box} 0
0 0 {box}
{elements}
 1  1  1  1  1

0.1 0.5 0.5
0.3 0.5 0.5
0.5 0.5 0.5
0.7 0.5 0.5
0.9 0.5 0.5
"""


def run_and_extract(d, box, elements):
    rundir = "rundir/"
    with open(f"{rundir}/POSCAR", "w") as fh:
        fh.write(get_poscar(d, box, elements))

    cmd = "/mnt/c/Users/guido/opt/xtb/6.4.0/bin/xtb --gfn 0 --sp --acc 1e-4 --strict --norestart --json POSCAR"
    output = (
        subprocess.check_output(shlex.split(cmd), cwd=rundir, stderr=subprocess.DEVNULL)
        .decode("utf8")
        .split("\n")
    )
    with open(f"{rundir}/xtbout.json") as fh:
        return json.load(fh)["electronic energy"]
    total = float([_ for _ in output if ":: total energy" in _][0].strip().split()[3])
    disp = float(
        [_ for _ in output if ":: dispersion energy" in _][0].strip().split()[3]
    )
    repulsion = float(
        [_ for _ in output if ":: repulsion energy" in _][0].strip().split()[3]
    )
    return total - disp - repulsion


def delta(d, box, elements1, elements2):
    return run_and_extract(d, box, elements1) - run_and_extract(d, box, elements2)


#%%
xs = np.append(np.arange(1, 3, 0.1), np.arange(3, 10, 1))
# ys1_5 = [delta(_, 5, UP, DN) for _ in xs]
# ys1_20 = [delta(_, 20, UP, DN) for _ in xs]
# ys1_50 = [delta(_, 50, UP, DN) for _ in xs]
# ys2_50 = [delta(_, 50, UP2, DN2) for _ in xs]
ys3_50 = [delta(_, 50, UP3, DN3) for _ in xs]


# region
plt.axhline(0, color="grey")

plt.plot(xs, ys1_5, "o-")
plt.plot(xs, ys1_20, "o-")
plt.plot(xs, ys1_50, "o-")
# region
plt.title(
    "xTB-GFN0 non-self-consistent energy difference\n between alchemical enantiomers of a 1D string in 3D PBC\nseparated by 50 $\AA$."
)
plt.axhline(0, color="grey")

plt.plot(xs, np.array(ys1_50) * 1000, "o-", label=f"{UP}<->{DN}")
plt.plot(xs, np.array(ys2_50) * 1000, "o-", label=f"{UP2}<->{DN2}")
plt.plot(xs, np.array(ys3_50) * 1000, "o-", label=f"{UP3}<->{DN3}")
plt.ylabel("$\Delta E$ [mHa]")
plt.xlabel("Atom spacing [$\mathrm{\AA}$]")
plt.legend()

# region
np.savetxt("out.txt", np.vstack((xs, ys1_50, ys2_50, ys3_50)).T)
# region
np.savetxt()
