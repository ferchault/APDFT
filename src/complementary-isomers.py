#!/usr/bin/env python
""" Generates all complementary isomers based on Carbon and their corresponding Gaussian input files."""

import argparse
import sys
import os
import itertools as it
import numpy as np
import apdft

parser = argparse.ArgumentParser(description="Generates complementary isomers.")
parser.add_argument("geometry", help="An XYZ file with the input molecule.", type=str)
parser.add_argument("method", help="A QM method.", choices=apdft.get_methods())
parser.add_argument(
    "basisset", help="A basis set. All of Basis Set Exchange supported."
)
parser.add_argument("directory", help="Output directory.")


def write_inputfile(path, contents):
    os.makedirs(path, exist_ok=True)
    with open(path + "/run.inp", "w") as fh:
        fh.write(contents)


if __name__ == "__main__":
    args = parser.parse_args()
    nuclear_numbers, coordinates = apdft.read_xyz(args.geometry)

    carbons = np.where(nuclear_numbers == 6)[0]
    calc = apdft.Calculator.GaussianCalculator(
        args.method, args.basisset, superimpose=False
    )

    # reference energy for comparison purposes
    thisnumbers = list(nuclear_numbers.copy())
    mid = calc.get_input(coordinates, thisnumbers, thisnumbers, None)
    path = "%s/mid" % (args.directory)
    write_inputfile(path, mid)

    # forward and backward complementary isomer candidates
    for combination in it.combinations(carbons, 2):
        thisnumbers = list(nuclear_numbers.copy())
        thisnumbers[combination[0]] = 7
        thisnumbers[combination[1]] = 5
        up = calc.get_input(coordinates, thisnumbers, thisnumbers, None)

        thisnumbers = list(nuclear_numbers.copy())
        thisnumbers[combination[0]] = 5
        thisnumbers[combination[1]] = 7
        dn = calc.get_input(coordinates, thisnumbers, thisnumbers, None)

        for direction, inputfile in {"up": up, "dn": dn}.items():
            path = "%s/%d-%d-%s" % (
                args.directory,
                combination[0],
                combination[1],
                direction,
            )
            write_inputfile(path, inputfile)
