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
import mpmath
import basis_set_exchange as bse


class Atom:
    """
    Class representing an atom.
    """

    def __init__(self, name, R, Z, basisZ):
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
        self.Z = Z
        self.basisZ = basisZ


class Basis:
    def __init__(self, basisset, atoms):
        self._basis = []
        for atom in atoms:
            db = bse.get_basis(basisset, int(atom.basisZ))

            db = db["elements"][str(int(atom.basisZ))]["electron_shells"]

            for shell in db:
                if shell["function_type"] != "gto":
                    raise ValueError()

                a = [mpmath.mpf(_) for _ in shell["exponents"]]
                for angmom, coeffs in zip(
                    shell["angular_momentum"], shell["coefficients"]
                ):
                    if angmom > 1:
                        raise ValueError()

                    d = [mpmath.mpf(_) for _ in coeffs]
                    if angmom == 0:
                        self._basis.append(
                            {
                                "R": atom.R,  #
                                "lx": 0,  #
                                "ly": 0,  #
                                "lz": 0,  #
                                "a": a,  #
                                "d": d,
                            }
                        )
                    if angmom == 1:
                        self._basis.append(
                            {
                                "R": atom.R,  #
                                "lx": 1,  #
                                "ly": 0,  #
                                "lz": 0,  #
                                "a": a,  #
                                "d": d,
                            }
                        )
                        self._basis.append(
                            {
                                "R": atom.R,  #
                                "lx": 0,  #
                                "ly": 1,  #
                                "lz": 0,  #
                                "a": a,  #
                                "d": d,
                            }
                        )
                        self._basis.append(
                            {
                                "R": atom.R,  #
                                "lx": 0,  #
                                "ly": 0,  #
                                "lz": 1,  #
                                "a": a,  #
                                "d": d,
                            }
                        )

        self.K = len(self._basis)

    def basis(self):
        return self._basis


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
