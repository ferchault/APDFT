import mpmath
import numpy as np
import numpy.linalg as la
import scipy.special as special
import scipy.special as spec
import scipy.integrate as quad
import tqdm


#-------------------------------------------------------------------------------
#basis.py
#-------------------------------------------------------------------------------

"""
    Copyright (C) 2015 Rocco Meli

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


class Atom:
    """
    Class representing an atom.
    """

    def __init__(self, name, R, Z, orbitals):
        """
        Initializer for ATOM

        INPUT:
            NAME: Name of the element
            R: Position (cartesian coordinates, atomic units)
            Z: Atomic charge
            ORBITALS: list of orbitals for this atom
        """

        self.name = name
        self.R = R
        self.orbitals = orbitals
        self.Z = Z


class STO3G:
    """
    STO-3G minimal basis set.
    """

    def __init__(self, atoms):
        """
        Initializer for STO3G

        INPUT:
            ATOMS: list of atoms

        Source

            Modern Quantum Chemistry
            Szabo and Ostlund
            Dover
            1989
        """

        # Exponential coefficients for the Gaussian orbitals
        self.zeta1 = {
            "H": mpmath.mp.mpf("1.24"),
            "He": mpmath.mp.mpf("2.0925"),
            "Li": mpmath.mp.mpf("2.69"),
            "Be": mpmath.mp.mpf("3.68"),
            "B": mpmath.mp.mpf("4.68"),
            "C": mpmath.mp.mpf("5.67"),
            "N": mpmath.mp.mpf("6.67"),
            "O": mpmath.mp.mpf("7.66"),
            "F": mpmath.mp.mpf("8.65"),
        }
        self.zeta2 = {
            "Li": mpmath.mp.mpf("0.75"),
            "Be": mpmath.mp.mpf("1.10"),
            "B": mpmath.mp.mpf("1.45"),
            "C": mpmath.mp.mpf("1.72"),
            "N": mpmath.mp.mpf("1.95"),
            "O": mpmath.mp.mpf("2.25"),
            "F": mpmath.mp.mpf("2.55"),
        }

        self.STO3G = []

        for a in atoms:  # For every atom
            for o in a.orbitals:  # For every atomic orbital
                if o == "1s":
                    a1 = 0.109818 * self.zeta1[a.name] ** 2
                    a2 = 0.405771 * self.zeta1[a.name] ** 2
                    a3 = 2.22766 * self.zeta1[a.name] ** 2
                    d1 = mpmath.mp.mpf("0.444635")
                    d2 = mpmath.mp.mpf("0.535328")
                    d3 = mpmath.mp.mpf("0.154329")

                    self.STO3G.append(
                        {
                            "AOn": a.name,  #
                            "AOt": o,  #
                            "R": a.R,  #
                            "lx": 0,  #
                            "ly": 0,  #
                            "lz": 0,  #
                            "a": (a1, a2, a3),  #
                            "d": (d1, d2, d3),
                        }
                    )

                if o == "2s":
                    a1 = 0.0751386 * self.zeta2[a.name] ** 2
                    a2 = 0.231031 * self.zeta2[a.name] ** 2
                    a3 = 0.994203 * self.zeta2[a.name] ** 2
                    d1 = mpmath.mp.mpf("0.700115")
                    d2 = mpmath.mp.mpf("0.399513")
                    d3 = mpmath.mp.mpf("-0.0999672")

                    self.STO3G.append(
                        {
                            "AOn": a.name,
                            "AOt": o,
                            "R": a.R,
                            "lx": 0,
                            "ly": 0,
                            "lz": 0,
                            "a": (a1, a2, a3),
                            "d": (d1, d2, d3),
                        }
                    )

                if o == "2p":
                    a1 = 0.0751386 * self.zeta2[a.name] ** 2
                    a2 = 0.231031 * self.zeta2[a.name] ** 2
                    a3 = 0.994203 * self.zeta2[a.name] ** 2
                    d1 = mpmath.mp.mpf("0.391957")
                    d2 = mpmath.mp.mpf("0.607684")
                    d3 = mpmath.mp.mpf("0.1559163")

                    self.STO3G.append(
                        {
                            "AOn": a.name,
                            "AOt": o,
                            "R": a.R,
                            "lx": 1,
                            "ly": 0,
                            "lz": 0,
                            "a": (a1, a2, a3),
                            "d": (d1, d2, d3),
                        }
                    )
                    self.STO3G.append(
                        {
                            "AOn": a.name,
                            "AOt": o,
                            "R": a.R,
                            "lx": 0,
                            "ly": 1,
                            "lz": 0,
                            "a": (a1, a2, a3),
                            "d": (d1, d2, d3),
                        }
                    )
                    self.STO3G.append(
                        {
                            "AOn": a.name,
                            "AOt": o,
                            "R": a.R,
                            "lx": 0,
                            "ly": 0,
                            "lz": 1,
                            "a": (a1, a2, a3),
                            "d": (d1, d2, d3),
                        }
                    )

            self.K = len(self.STO3G)

    def basis(self):
        """
        Return the basis set.
        """

        return self.STO3G

    def info(self):
        """
        Print informations about the bais set.
        """
        print("########################")
        print("STO-3G MINIMAL BASIS SET")
        print("########################\n")

        for b in self.STO3G:
            print(b["AOn"] + " orbital:")
            print("   " + b["AOt"] + ":")
            print("      R = ", b["R"])
            print("      lx = " + str(b["lx"]))
            print("      ly = " + str(b["ly"]))
            print("      lz = " + str(b["lz"]))
            print("      alpha = ", b["a"])
            print("      d = ", b["d"], "\n")

#-------------------------------------------------------------------------------
#integrals.py
#-------------------------------------------------------------------------------

def gaussian_product(aa, bb, Ra, Rb):
    """
    Gaussian produc theorem.

    INPUT:
        AA: Exponential coefficient of Gaussian 1
        BB: Exponential coefficient of Gaussian 2
        RA: Center of Gaussian 1
        RB: Center of Gaussian 2
    OUTPUT:
        R: Gaussian product center
        C: Gaussian product coefficient

    Source:
        Modern Quantum Chemistry
        Szabo and Ostlund
        Dover
        1989
    """

    # Transform centers in Numpy arrays
    Ra = np.asarray(Ra)
    Rb = np.asarray(Rb)

    # Compute Gaussian product center
    R = (aa * Ra + bb * Rb) / (aa + bb)

    # Compute Gaussian product coefficient
    c = np.dot(Ra - Rb, Ra - Rb)
    c *= -aa * bb / (aa + bb)
    exp = np.vectorize(mpmath.exp)
    c = exp(c)

    return R, c


def norm(ax, ay, az, aa):
    """
    General cartesian Gaussian normalization factor.

    INPUT:
        AX: Angular momentum lx
        AY: Angular momentum ly
        AZ: Angular momentum lz
        AA: Gaussian exponential coefficient
    OUTPUT:
        N: Normalization coefficient (to be multiplied with the Gaussian)

    Source:
        Handbook of Computational Chemistry
        David Cook
        Oxford University Press
        1998
    """

    # Compute normalization coefficient
    N = (2 * aa / np.pi) ** (3.0 / 4.0)
    N *= (4 * aa) ** ((ax + ay + az) / 2)
    N /= np.sqrt(
        special.factorial2(2 * ax - 1)
        * special.factorial2(2 * ay - 1)
        * special.factorial2(2 * az - 1)
    )

    return N


def Sxyz(a, b, aa, bb, Ra, Rb, R):
    """
    Compute overlap integral between two unnormalized Cartesian gaussian functions along one direction.

    INPUT:
        A: Angular momentum along the chosen direction for Gaussian 1
        B: Angular momentum along the chosen direction for Gaussian 2
        AA: Exponential coefficient for Gaussian 1
        BB: Exponential coefficient for Gaussian 2
        RA: Coordinate (along chosen direction) of the center of Gaussian 1
        RB: Coordinate (along chosen direction) of the center of Gaussian 2
        R: Coordinate (along chosen direction) of the center of the product of the two gaussians
    OUTPUT:
        S: Overlap of the two gaussians along the chosen direction

    Source:
        The Mathematica Journal
        Evaluation of Gaussian Molecular Integrals
        I. Overlap Integrals
        Minhhuy Hô and Julio Manuel Hernández-Pérez
    """

    S = 0

    for i in range(a + 1):
        for j in range(b + 1):
            if (i + j) % 2 == 0:
                tmp = special.comb(a, i, exact=True)
                tmp *= special.comb(b, j, exact=True)
                tmp *= special.factorial2(i + j - 1, exact=True)
                tmp /= (2.0 * (aa + bb)) ** ((i + j) / 2.0)
                tmp *= (R - Ra) ** (a - i)
                tmp *= (R - Rb) ** (b - j)

                S += tmp

    return S


def overlap(ax, ay, az, bx, by, bz, aa, bb, Ra, Rb):
    """
    Compute overlap integral between two Cartesian gaussian functions.

    INPUT:
        AX: Angular momentum lx for Gaussian 1
        AY: Angular momentum ly for Gaussian 1
        AZ: Angular momentum lz for Gaussian 1
        AA: Exponential coefficient for Gaussian 1
        BX: Angular momentum lx for Gaussian 2
        BY: Angular momentum ly for Gaussian 2
        BZ: Angular momentum lz for Gaussian 2
        BB: Exponential coefficient for Gaussian 2
        RA: Center of Gaussian 1
        RB: Center of Gaussian 2
    OUTPUT:
        S: Overlap of the two gaussians

    Source:
        The Mathematica Journal
        Evaluation of Gaussian Molecular Integrals
        I. Overlap Integrals
        Minhhuy Hô and Julio Manuel Hernández-Pérez
    """

    # Compute gaussian product center and coefficient
    R, c = gaussian_product(aa, bb, Ra, Rb)

    # Compute normalization factors for the two gaussians
    Na = norm(ax, ay, az, aa)
    Nb = norm(bx, by, bz, bb)

    S = 1
    S *= Sxyz(ax, bx, aa, bb, Ra[0], Rb[0], R[0])  # Overlap along x
    S *= Sxyz(ay, by, aa, bb, Ra[1], Rb[1], R[1])  # Overlap along y
    S *= Sxyz(az, bz, aa, bb, Ra[2], Rb[2], R[2])  # Overlap along z
    S *= Na * Nb * c  # Product coefficient and normalization
    S *= (np.pi / (aa + bb)) ** (3.0 / 2.0)  # Normalization

    return S


def kinetic(ax, ay, az, bx, by, bz, aa, bb, Ra, Rb):
    """
    Compute kinetic integral between two Cartesian gaussian functions.

    INPUT:
        AX: Angular momentum lx for Gaussian 1
        AY: Angular momentum ly for Gaussian 1
        AZ: Angular momentum lz for Gaussian 1
        AA: Exponential coefficient for Gaussian 1
        BX: Angular momentum lx for Gaussian 2
        BY: Angular momentum ly for Gaussian 2
        BZ: Angular momentum lz for Gaussian 2
        BB: Exponential coefficient for Gaussian 2
        RA: Center of Gaussian 1
        RB: Center of Gaussian 2
    OUTPUT:
        K: Kinetic integral between the two gaussians

    Source:
        The Mathematica Journal
        Evaluation of Gaussian Molecular Integrals
        II. Kinetic-Energy Integrals
        Minhhuy Hô and Julio Manuel Hernández-Pérez
    """

    R, c = gaussian_product(aa, bb, Ra, Rb)

    def Kxyz(ac, a1, a2, bc, b1, b2, aa, bb, Ra, Rb, Ra1, Rb1, Ra2, Rb2, Rc, R1, R2):
        """
            Compute kinetic integral between two Cartesian gaussian functions along one direction.

            INPUT:
                AC: Component of angular momentum for Gaussian 1 along direction of interest
                A1: Component of angular momentum for Gaussian 1 along second direction
                A2: Component of angular momentum for Gaussian 1 along third direction
                BC: Component of angular momentum for Gaussian 2 along direction of interest
                B1: Component of angular momentum for Gaussian 2 along second direction
                B2: Component of angular momentum for Gaussian 2 along third direction
                AA: Exponential coefficient for Gaussian 1
                BB: Exponential coefficient for Gaussian 2
                RA: Component of the center of Gaussian 1 along direction of interest
                RB: Component of the center of Gaussian 2 along direction of interest
                RA1: Component of the center of Gaussian 1 along second direction
                RA2: Component of the center of Gaussian 1 along third direction
                RB1: Component of the center of Gaussian 2 along second direction
                RB2: Component of the center of Gaussian 2 along third direction
            OUTPUT:
                KC: Kinetic integral between two gaussians along chosen direction

            Source:
                The Mathematica Journal
                Evaluation of Gaussian Molecular Integrals
                II. Kinetic-Energy Integrals
                Minhhuy Hô and Julio Manuel Hernández-Pérez
        """

        kc = 0
        kc += ac * bc * Sxyz(ac - 1, bc - 1, aa, bb, Ra, Rb, Rc)
        kc += -2 * aa * bc * Sxyz(ac + 1, bc - 1, aa, bb, Ra, Rb, Rc)
        kc += -2 * ac * bb * Sxyz(ac - 1, bc + 1, aa, bb, Ra, Rb, Rc)
        kc += 4 * aa * bb * Sxyz(ac + 1, bc + 1, aa, bb, Ra, Rb, Rc)
        kc *= 0.5

        Kc = 1
        Kc *= c * (np.pi / (aa + bb)) ** (3.0 / 2.0) * kc
        Kc *= Sxyz(a1, b1, aa, bb, Ra1, Rb1, R1)
        Kc *= Sxyz(a2, b2, aa, bb, Ra2, Rb2, R2)

        return Kc

    # Cyclic permutation of the entries
    Kx = Kxyz(
        ax,
        ay,
        az,
        bx,
        by,
        bz,
        aa,
        bb,
        Ra[0],
        Rb[0],
        Ra[1],
        Rb[1],
        Ra[2],
        Rb[2],
        R[0],
        R[1],
        R[2],
    )  # Kinetic integral along x
    Ky = Kxyz(
        ay,
        az,
        ax,
        by,
        bz,
        bx,
        aa,
        bb,
        Ra[1],
        Rb[1],
        Ra[2],
        Rb[2],
        Ra[0],
        Rb[0],
        R[1],
        R[2],
        R[0],
    )  # Kinetic integral along y
    Kz = Kxyz(
        az,
        ax,
        ay,
        bz,
        bx,
        by,
        aa,
        bb,
        Ra[2],
        Rb[2],
        Ra[0],
        Rb[0],
        Ra[1],
        Rb[1],
        R[2],
        R[0],
        R[1],
    )  # Kinetic integral along z

    Na = norm(ax, ay, az, aa)  # Normalization factor for Gaussian 1
    Nb = norm(bx, by, bz, bb)  # Normalization factor for Gaussian 2

    K = (Kx + Ky + Kz) * Na * Nb  # Normalization of total kinetic energy integral

    return K


def f(j, l, m, a, b):
    """
    Expansion coefficient f.

    Source:
        Handbook of Computational Chemistry
        David Cook
        Oxford University Press
        1998
    """

    f = 0

    for k in range(max(0, j - m), min(j, l) + 1):
        tmp = 1
        tmp *= spec.binom(l, k)
        tmp *= spec.binom(m, j - k)
        tmp *= a ** (l - k)
        tmp *= b ** (m + k - j)

        f += tmp

    return f


def F(nu, x):
    """
    Boys function.

    INPUT:
        NU: Boys function index
        X: Boys function variable

    OUTPUT:
        FF: Value of the Boys function for index NU evaluated at X

    Source:
        Evaluation of the Boys Function using Analytical Relations
        I. I. Guseinov and B. A. Mamedov
        Journal of Mathematical Chemistry
        2006
    """

    if x < 1e-8:
        # Taylor expansion for argument close or equal to zero (avoid division by zero)
        ff = 1 / (2 * nu + 1) - x / (2 * nu + 3)
    else:
        # Evaluate Boys function with incomplete and complete Gamma functions
        ff = 0.5 / x ** (nu + 0.5) * spec.gamma(nu + 0.5) * mpmath.gammainc(nu + 0.5, x)

    return ff


def nuclear(ax, ay, az, bx, by, bz, aa, bb, Ra, Rb, Rn, Zn):
    """
    Compute nuclear-electron interaction integrals.

    INPUT:
    AX,AY,AZ: Angular momentum components for the first Gaussian.
    BX,BY,BZ: Angular momentum components for the second Gaussian.
    AA: Exponential coefficient for the first Gaussian.
    BB: Exponential coefficient for the second Gaussian.
    RA: Center of the first Gaussian.
    RB: Center of the second Gaussian.
    RN: Nuclear coordinates.
    ZN: Nuclear charge.

    Source:

        Handbook of Computational Chemistry
        David Cook
        Oxford University Press
        1998
    """

    Vn = 0

    # Intermediate variable
    g = aa + bb
    eps = 1.0 / (4 * g)

    Rp, c = gaussian_product(aa, bb, Ra, Rb)  # Gaussian product

    def A(l, r, i, l1, l2, Ra, Rb, Rc, Rp):
        """
        Expansion coefficient A.

        Source:
            Handbook of Computational Chemistry
            David Cook
            Oxford University Press
            1998
        """

        A = 1
        A *= (-1) ** (l)
        A *= f(l, l1, l2, Rp - Ra, Rp - Rb)
        A *= (-1) ** i
        A *= special.factorial(l, exact=True)
        A *= (Rp - Rc) ** (l - 2 * r - 2 * i)
        A *= eps ** (r + i)
        A /= special.factorial(r, exact=True)
        A /= special.factorial(i, exact=True)
        A /= special.factorial(l - 2 * r - 2 * i, exact=True)

        return A

    for l in range(0, ax + bx + 1):
        for r in range(0, int(l / 2) + 1):
            for i in range(0, int((l - 2 * r) / 2) + 1):
                Ax = A(l, r, i, ax, bx, Ra[0], Rb[0], Rn[0], Rp[0])

                for m in range(0, ay + by + 1):
                    for s in range(0, int(m / 2) + 1):
                        for j in range(0, int((m - 2 * s) / 2) + 1):
                            Ay = A(m, s, j, ay, by, Ra[1], Rb[1], Rn[1], Rp[1])

                            for n in range(0, az + bz + 1):
                                for t in range(0, int(n / 2) + 1):
                                    for k in range(0, int((n - 2 * t) / 2) + 1):
                                        Az = A(
                                            n, t, k, az, bz, Ra[2], Rb[2], Rn[2], Rp[2]
                                        )

                                        nu = (
                                            l + m + n - 2 * (r + s + t) - (i + j + k)
                                        )  # Index of Boys function

                                        ff = F(
                                            nu, g * np.dot(Rp - Rn, Rp - Rn)
                                        )  # Boys function

                                        Vn += Ax * Ay * Az * ff

    # Compute normalization
    Na = norm(ax, ay, az, aa)
    Nb = norm(bx, by, bz, bb)

    Vn *= -Zn * Na * Nb * c * 2 * np.pi / g

    return Vn


def electronic(
    ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, aa, bb, cc, dd, Ra, Rb, Rc, Rd
):
    """
    Compute electron-electron interaction integrals.

    INPUT:
        AX,AY,AZ: Angular momentum components for the first Gaussian.
        BX,BY,BZ: Angular momentum components for the second Gaussian.
        CX,CY,CZ: Angular momentum components for the third Gaussian.
        DX,DY,DZ: Angular momentum components for the fourth Gaussian.
        AA: Exponential coefficient for the first Gaussian.
        BB: Exponential coefficient for the second Gaussian.
        CC: Exponential coefficient for the third Gaussian.
        DD: Exponential coefficient for the fourth Gaussian.
        RA: Center of the first Gaussian.
        RB: Center of the second Gaussian.
        RC: Center of the third Gaussian.
        RD: Center of the fourth Gaussian.
    OUTPUT:
        G: Electron-electron integral

    Source:

        Handbook of Computational Chemistry
        David Cook
        Oxford University Press
        1998

        ERRATA (the original formula is WRONG)!
            http://spider.shef.ac.uk/
    """

    G = 0

    # Intermediate variable
    g1 = aa + bb
    g2 = cc + dd

    # Compute gaussian products
    Rp, c1 = gaussian_product(aa, bb, Ra, Rb)
    Rq, c2 = gaussian_product(cc, dd, Rc, Rd)

    delta = 1 / (4 * g1) + 1 / (4 * g2)

    def theta(l, l1, l2, a, b, r, g):
        """
        Expansion coefficient theta.

        Source:
            Handbook of Computational Chemistry
            David Cook
            Oxford University Press
            1998
        """

        t = 1
        t *= f(l, l1, l2, a, b)
        t *= special.factorial(l, exact=True)
        t *= g ** (r - l)
        t /= special.factorial(r, exact=True) * special.factorial(l - 2 * r, exact=True)

        return t

    def B(l, ll, r, rr, i, l1, l2, Ra, Rb, Rp, g1, l3, l4, Rc, Rd, Rq, g2):
        """
        Expansion coefficient B.

        Source:
            Handbook of Computational Chemistry
            David Cook
            Oxford University Press
            1998
        """

        b = 1
        b *= (-1) ** (l) * theta(l, l1, l2, Rp - Ra, Rp - Rb, r, g1)
        b *= theta(ll, l3, l4, Rq - Rc, Rq - Rd, rr, g2)
        b *= (-1) ** i * (2 * delta) ** (2 * (r + rr))
        b *= special.factorial(l + ll - 2 * r - 2 * rr, exact=True)
        b *= delta ** i * (Rp - Rq) ** (l + ll - 2 * (r + rr + i))

        tmp = 1
        tmp *= (4 * delta) ** (l + ll) * special.factorial(i, exact=True)
        tmp *= special.factorial(l + ll - 2 * (r + rr + i), exact=True)

        b /= tmp

        return b

    for l in range(0, ax + bx + 1):
        for r in range(0, int(l / 2) + 1):
            for ll in range(0, cx + dx + 1):
                for rr in range(0, int(ll / 2) + 1):
                    for i in range(0, int((l + ll - 2 * r - 2 * rr) / 2) + 1):
                        Bx = B(
                            l,
                            ll,
                            r,
                            rr,
                            i,
                            ax,
                            bx,
                            Ra[0],
                            Rb[0],
                            Rp[0],
                            g1,
                            cx,
                            dx,
                            Rc[0],
                            Rd[0],
                            Rq[0],
                            g2,
                        )

                        for m in range(0, ay + by + 1):
                            for s in range(0, int(m / 2) + 1):
                                for mm in range(0, cy + dy + 1):
                                    for ss in range(0, int(mm / 2) + 1):
                                        for j in range(
                                            0, int((m + mm - 2 * s - 2 * ss) / 2) + 1
                                        ):
                                            By = B(
                                                m,
                                                mm,
                                                s,
                                                ss,
                                                j,
                                                ay,
                                                by,
                                                Ra[1],
                                                Rb[1],
                                                Rp[1],
                                                g1,
                                                cy,
                                                dy,
                                                Rc[1],
                                                Rd[1],
                                                Rq[1],
                                                g2,
                                            )

                                            for n in range(0, az + bz + 1):
                                                for t in range(0, int(n / 2) + 1):
                                                    for nn in range(0, cz + dz + 1):
                                                        for tt in range(
                                                            0, int(nn / 2) + 1
                                                        ):
                                                            for k in range(
                                                                0,
                                                                int(
                                                                    (
                                                                        n
                                                                        + nn
                                                                        - 2 * t
                                                                        - 2 * tt
                                                                    )
                                                                    / 2
                                                                )
                                                                + 1,
                                                            ):
                                                                Bz = B(
                                                                    n,
                                                                    nn,
                                                                    t,
                                                                    tt,
                                                                    k,
                                                                    az,
                                                                    bz,
                                                                    Ra[2],
                                                                    Rb[2],
                                                                    Rp[2],
                                                                    g1,
                                                                    cz,
                                                                    dz,
                                                                    Rc[2],
                                                                    Rd[2],
                                                                    Rq[2],
                                                                    g2,
                                                                )

                                                                nu = (
                                                                    l
                                                                    + ll
                                                                    + m
                                                                    + mm
                                                                    + n
                                                                    + nn
                                                                    - 2
                                                                    * (
                                                                        r
                                                                        + rr
                                                                        + s
                                                                        + ss
                                                                        + t
                                                                        + tt
                                                                    )
                                                                    - (i + j + k)
                                                                )

                                                                ff = F(
                                                                    nu,
                                                                    np.dot(
                                                                        Rp - Rq, Rp - Rq
                                                                    )
                                                                    / (4.0 * delta),
                                                                )

                                                                G += Bx * By * Bz * ff

    # Compute normalization
    Na = norm(ax, ay, az, aa)
    Nb = norm(bx, by, bz, bb)
    Nc = norm(cx, cy, cz, cc)
    Nd = norm(dx, dy, dz, dd)

    G *= (
        Na
        * Nb
        * Nc
        * Nd
        * c1
        * c2
        * 2
        * np.pi ** 2
        / (g1 * g2)
        * np.sqrt(np.pi / (g1 + g2))
    )

    return G


def EE_list(basis):
    """
    Multidimensional array of two-electron integrals.

    INPUT:
        BASIS: Basis set
    OUTPUT:
        EE: list of two-electron integrals, with indices (i,j,k,l)
    """

    # Size of the basis set
    K = basis.K

    # List of basis functions
    B = basis.basis()

    EE = mpmath.matrix(K, K, K, K)

    with tqdm.tqdm(desc="2e integrals", total=len(B) ** 4) as pbar:
        for i, b1 in enumerate(B):
            for j, b2 in enumerate(B):
                for k, b3 in enumerate(B):
                    for l, b4 in enumerate(B):

                        pbar.update(1)

                        for a1, d1 in zip(b1["a"], b1["d"]):
                            for a2, d2 in zip(b2["a"], b2["d"]):
                                for a3, d3 in zip(b3["a"], b3["d"]):
                                    for a4, d4 in zip(b4["a"], b4["d"]):
                                        # Basis functions centers
                                        R1 = b1["R"]
                                        R2 = b2["R"]
                                        R3 = b3["R"]
                                        R4 = b4["R"]

                                        # Basis functions angular momenta
                                        ax = b1["lx"]
                                        ay = b1["ly"]
                                        az = b1["lz"]

                                        # Basis functions angular momenta
                                        bx = b2["lx"]
                                        by = b2["ly"]
                                        bz = b2["lz"]

                                        # Basis functions angular momenta
                                        cx = b3["lx"]
                                        cy = b3["ly"]
                                        cz = b3["lz"]

                                        # Basis functions angular momenta
                                        dx = b4["lx"]
                                        dy = b4["ly"]
                                        dz = b4["lz"]

                                        tmp = 1
                                        tmp *= d1.conjugate() * d2.conjugate()
                                        tmp *= d3 * d4
                                        tmp *= electronic(
                                            ax,
                                            ay,
                                            az,
                                            bx,
                                            by,
                                            bz,
                                            cx,
                                            cy,
                                            cz,
                                            dx,
                                            dy,
                                            dz,
                                            a1,
                                            a2,
                                            a3,
                                            a4,
                                            R1,
                                            R2,
                                            R3,
                                            R4,
                                        )

                                        EE[i, j, k, l] += tmp

    return EE


def print_EE_list(basis, ee):
    """
    Print list of electron-electron integrals.

    INPUT:
        EE: list of electron-electron integrals (computed by EE_LIST function)
    """

    K = basis.K

    for i in range(K):
        for j in range(K):
            for k in range(K):
                for l in range(K):
                    print(
                        "({0},{1},{2},{3})  {4}".format(
                            i + 1, j + 1, k + 1, l + 1, ee[i, j, k, l]
                        )
                    )

#-------------------------------------------------------------------------------
#matrices.py
#-------------------------------------------------------------------------------

def S_overlap(basis):
    """
    Compute overlap matrix S.

    INPUT:
        BASIS: basis set
    OUTPUT:
        S: Overlap matrix
    """

    # Size of the basis set
    K = basis.K

    # List of basis functions
    B = basis.basis()

    S = mpmath.matrix(K, K)

    for i, b1 in enumerate(B):
        for j, b2 in enumerate(B):
            for a1, d1 in zip(b1["a"], b1["d"]):
                for a2, d2 in zip(b2["a"], b2["d"]):
                    R1 = b1["R"]
                    R2 = b2["R"]

                    tmp = d1.conjugate() * d2
                    tmp *= overlap(
                        b1["lx"],
                        b1["ly"],
                        b1["lz"],
                        b2["lx"],
                        b2["ly"],
                        b2["lz"],
                        a1,
                        a2,
                        R1,
                        R2,
                    )

                    S[i, j] += tmp

    return S


def X_transform(S):
    """
    Compute the transformation matrix X using canonical orthogonalization.

    INPUT:
        S: Overlap matrix
    OUTPUT:
        X: Transformation matrix

    Source:
        Modern Quantum Chemistry
        Szabo and Ostlund
        Dover
        1989
    """

    s, U = mpmath.eig(S)

    s = np.diag([mpmath.power(_, (-1.0 / 2.0)) for _ in s])

    # X = np.dot(U, s)
    X = U * mpmath.matrix(s)

    return X


def T_kinetic(basis):
    """
    Compute kinetic matrix T.

    INPUT:
        BASIS: basis set
    OUTPUT:
        T: Kinetic matrix
    """
    # Size of the basis set
    K = basis.K

    # List of basis functions
    B = basis.basis()

    T = mpmath.matrix(K, K)

    for i, b1 in enumerate(B):
        for j, b2 in enumerate(B):
            for a1, d1 in zip(b1["a"], b1["d"]):
                for a2, d2 in zip(b2["a"], b2["d"]):
                    R1 = b1["R"]
                    R2 = b2["R"]

                    tmp = d1.conjugate() * d2
                    tmp *= kinetic(
                        b1["lx"],
                        b1["ly"],
                        b1["lz"],
                        b2["lx"],
                        b2["ly"],
                        b2["lz"],
                        a1,
                        a2,
                        R1,
                        R2,
                    )

                    T[i, j] += tmp

    return T


def V_nuclear(basis, atom):
    """
    Compute nuclear-electron potential energy matrix Vn.

    INPUT:
        BASIS: basis set
        ATOM: atom specifications (position and charge)
    OUTPUT:
        VN: Nuclear-attraction matrix for atom ATOM
    """
    # Size of the basis set
    K = basis.K

    # List of basis functions
    B = basis.basis()

    # Nuclear coordinates
    Rn = atom.R

    # Nuclear charge
    Zn = atom.Z

    Vn = mpmath.matrix(K, K)

    for i, b1 in enumerate(B):
        for j, b2 in enumerate(B):
            for a1, d1 in zip(b1["a"], b1["d"]):
                for a2, d2 in zip(b2["a"], b2["d"]):
                    R1 = b1["R"]
                    R2 = b2["R"]

                    tmp = d1.conjugate() * d2
                    tmp *= nuclear(
                        b1["lx"],
                        b1["ly"],
                        b1["lz"],
                        b2["lx"],
                        b2["ly"],
                        b2["lz"],
                        a1,
                        a2,
                        R1,
                        R2,
                        Rn,
                        Zn,
                    )

                    Vn[i, j] += tmp

    return Vn


def H_core(basis, molecule):
    """
    Compute core Hamiltonian (sum of T and all the VN)

    INPUT:
        BASIS: basis set
        MOLECULE: molecule, collection of atom objects
    OUTPUT:
        (T + VN): Core Hamitlonian
    """
    T = T_kinetic(basis)

    # print("Kinetic energy")
    # print(T)

    # Size of the basis set
    K = basis.K

    Vn = mpmath.matrix(K, K)

    Vnn = mpmath.matrix(K, K)

    for atom in molecule:
        Vnn = V_nuclear(basis, atom)

        # print("Nuclear attraction Vn")
        # print(Vnn)

        Vn += Vnn

    # print("Total nuclear attraction matrix")
    # print(Vn)

    return T + Vn


def P_density(C, N):
    """
    Compute dansity matrix.

    INPUT:
        C: Matrix of coefficients
        N: Number of electrons
    OUTPUT:
        P: density matrix

    Source:
        Modern Quantum Chemistry
        Szabo and Ostlund
        Dover
        1989
    """

    # Size of the basis set
    K = C.shape[0]

    P = mpmath.matrix(K, K)

    for i in range(K):
        for j in range(K):
            for k in range(int(N / 2)):  # TODO Only for RHF
                P[i, j] += 2 * C[i, k] * C[j, k].conjugate()

    return P


def G_ee(basis, molecule, P, ee):
    """
    Compute core Hamiltonian matrix.

    INPUT:
        BASIS: Basis set.
        MOLECULE: Collection of atoms
        P: Density matrix
        EE: Two-electron integrals
    OUTPUT:
        G: Electron-electron interaction matrix
    """

    # Size of the basis set
    K = basis.K

    G = mpmath.matrix(K, K)

    for i in range(K):
        for j in range(K):
            for k in range(K):
                for l in range(K):
                    G[i, j] += P[k, l] * (ee[i, j, k, l] - 0.5 * ee[i, l, k, j])

    return G

#-------------------------------------------------------------------------------
#RHF.py
#-------------------------------------------------------------------------------


def to_np(matrix):
    q = np.array(matrix.tolist())
    if q.shape == (len(q), 1):
        return q.reshape(-1)
    return q


def from_np(array):
    return mpmath.matrix(array.tolist())


def RHF_step(basis, molecule, N, H, X, P_old, ee, verbose=False):
    """
    Restricted Hartree-Fock self-consistent field setp.

    INPUT:
        BASIS: basis set
        MOLECULE: molecule, collection of atom object
        N: Number of electrons
        H: core Hamiltonian
        X: tranformation matrix
        P_OLD: Old density matrix
        EE: List of electron-electron Integrals
        VERBOSE: verbose flag (set True to print everything on screen)
    """

    if verbose:
        print("\nDensity matrix P:")
        print(P_old)

    G = G_ee(basis, molecule, P_old, ee)  # Compute electron-electron interaction matrix

    if verbose:
        print("\nG matrix:")
        print(G)

    F = to_np(H) + to_np(G)  # Compute Fock matrix

    if verbose:
        print("\nFock matrix:")
        print(F)

    Fx = np.dot(
        to_np(X).T, np.dot(F, to_np(X))
    )  # Compute Fock matrix in the orthonormal basis set (S=I in this set)

    if verbose:
        print("\nFock matrix in orthogonal orbital basis:")
        print(Fx)

    e, Cx = mpmath.eigh(
        from_np(Fx)
    )  # Compute eigenvalues and eigenvectors of the Fock matrix

    # Sort eigenvalues from smallest to highest (needed to compute P correctly)
    idx = to_np(e).argsort()
    e = to_np(e)[idx]
    Cx = to_np(Cx)[:, idx]

    if verbose:
        print("\nCoefficients in orthogonal orbital basis:")
        print(Cx)

    e = np.diag(e)  # Extract orbital energies as vector

    if verbose:
        print("\nEnergies in orthogonal orbital basis:")
        print(e)

    C = np.dot(
        to_np(X), to_np(Cx)
    )  # Transform coefficient matrix in the orthonormal basis to the original basis

    if verbose:
        print("\nCoefficients:")
        print(C)

    Pnew = P_density(C, N)  # Compute the new density matrix

    return Pnew, F, e


def delta_P(P_old, P_new):
    """
    Compute the difference between two density matrices.

    INTPUT:
        P_OLD: Olde density matrix
        P_NEW: New density matrix
    OUTPUT:
        DELTA: difference between the two density matrices

    Source:
        Modern Quantum Chemistry
        Szabo and Ostlund
        Dover
        1989
    """
    delta = 0

    n = to_np(P_old).shape[0]

    for i in range(n):
        for j in range(n):
            delta += (P_old[i, j] - P_new[i, j]) ** 2

    return (delta / 4.0) ** (0.5)


def energy_el(P, F, H):
    """
    Compute electronic energy.

    INPUT:
        P: Density matrix
        F: Fock matrix
        H: Core Hamiltonian

    Source:
        Modern Quantum Chemistry
        Szabo and Ostlund
        Dover
        1989
    """

    # Size of the basis set
    K = to_np(P).shape[0]

    E = 0

    for i in range(K):
        for j in range(K):
            E += 0.5 * P[i, j] * (H[i, j] + F[i, j])

    return E


def energy_n(molecule):
    """
    Compute nuclear energy (classical nucleus-nucleus repulsion)

    INPUT:
        MOLECULE: molecule, as a collection of atoms
    OUTPUT:
        ENERGY_N: Nuclear energy
    """

    en = 0

    for i in range(len(molecule)):
        for j in range(i + 1, len(molecule)):
            # Select atoms from molecule
            atomi = molecule[i]
            atomj = molecule[j]

            # Extract distance from atom
            Ri = np.asarray(atomi.R)
            Rj = np.asarray(atomj.R)

            en += atomi.Z * atomj.Z / la.norm(Ri - Rj)

    return en


def energy_tot(P, F, H, molecule):
    """
    Compute total energy (electronic plus nuclear).

    INPUT:
        P: Density matrix
        F: Fock matrix
        H: Core Hamiltonian
        MOLECULE: molecule, as a collection of atoms
    OUTPUT:
        ENERGY_TOT: total energy
    """
    return energy_el(P, F, H) + energy_n(molecule)


#-------------------------------------------------------------------------------
#molecules.py
#-------------------------------------------------------------------------------

#mol, N = MAGtoMole(water)

# H2
H2 = [
    Atom("H", (mpmath.mpf(0), mpmath.mpf(0), mpmath.mpf(0)), mpmath.mpf("2"), ["1s"]),
    Atom(
        "H", (mpmath.mpf(0), mpmath.mpf(0), mpmath.mpf("1.4")), mpmath.mpf("0"), ["1s"]
    ),
]

# HeH+
HeH = [Atom("He", (0, 0, 1.4632), 2, ["1s"]), Atom("H", (0, 0, 0), 1, ["1s"])]

# He
He = [Atom("He", (0, 0, 0), 2, ["1s"])]

# B
Be = [Atom("Be", (0, 0, 0), 4, ["1s", "2s"])]

# N2
N2 = [
    Atom("N", (0, 0, 0), 7, ["1s", "2s", "2p"]),
    Atom("N", (0, 0, 2.074), 7, ["1s", "2s", "2p"]),
]

# HF
HF = [Atom("H", (0, 0, 0), 1, ["1s"]), Atom("F", (0, 0, 1.807), 9, ["1s", "2s", "2p"])]

# B
O = [Atom("O", (0, 0, 0), 8, ["1s", "2s", "2p"])]
# H2O
# Szabo and Ostlund
H2O = [
    Atom("H", (1.809 * np.sin(104.52 / 180 * np.pi / 2), 0, 0), 1, ["1s"]),
    Atom("H", (-1.809 * np.sin(104.52 / 180 * np.pi / 2), 0, 0), 1, ["1s"]),
    Atom("O", (0, 1.809 * np.cos(104.52 / 180 * np.pi / 2), 0), 8, ["1s", "2s", "2p"]),
]
# Mathematica Journal
# H2O = [ Atom("H",(0,1.43233673,-0.96104039),1,["1s"]),
#        Atom("H",(0,-1.43233673,-0.96104039),1,["1s"]),
#        Atom("O",(0,0,0.24026010),8,["1s","2s","2p"])]


def energy_tot_arbprec(mol, N):
    bs = STO3G(mol) # Basis set
    #N = 10  # Number of electrons

    maxiter = 100  # Maximal number of iteration

    verbose = False  # Print each SCF step

    ###########################
    ###########################
    ###########################

    # Basis set size
    K = bs.K

    # print("Computing overlap matrix S...")
    S = S_overlap(bs)

    if verbose:
        print(S)

    # print("Computing orthogonalization matrix X...")
    X = X_transform(S)

    if verbose:
        print(X)

    # print("Computing core Hamiltonian...")
    Hc = H_core(bs, mol)

    if verbose:
        print(Hc)

    # print("Computing two-electron integrals...")
    ee = EE_list(bs)

    if verbose:
        print_EE_list(bs, ee)

    Pnew = mpmath.matrix(K, K)
    P = mpmath.matrix(K, K)

    converged = False

    iter = 1
    while not converged:
        # print("\n\n\n#####\nSCF cycle " + str(iter) + ":")
        # print("#####")

        Pnew, F, E = RHF_step(bs, mol, N, Hc, X, P, ee, verbose)  # Perform an SCF step

        # Print results of the SCF step
        e = energy_tot(P, F, Hc, mol)
        # print("   Orbital energies:")
        # print("   ", np.diag(E))

        # Check convergence of the SCF cycle
        dp = delta_P(P, Pnew)
        print(f"{iter:>5} {mpmath.nstr(dp, 5, strip_zeros=False):10} {e}")
        if dp < mpmath.mpf(f"1e-{mpmath.mp.dps-5}"):
            converged = True
            print(
                "\n\n\nTOTAL ENERGY:", energy_tot(P, F, Hc, mol)
            )  # Print final, total energy
            return energy_tot(P, F, Hc, mol)

        P = Pnew

        iter += 1
