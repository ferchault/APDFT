#!/usr/bin/env python
import argparse
import mqm

parser = argparse.ArgumentParser(description='QM calculations on multiple systems.')
parser.add_argument('geometry', help='An XYZ file with the input molecule.', type=str)
parser.add_argument('method', help='A QM method.', choices=mqm.get_methods())
parser.add_argument('basisset', help='A basis set. All of Basis Set Exchange supported.')

args = parser.parse_args()

calculator = mqm.Calculator.GaussianCalculator()
nuclear_numbers, coordinates = mqm.read_xyz(args.geometry)

derivatives = mqm.Derivatives.DerivativeFolders(calculator, 2, nuclear_numbers, coordinates)
derivatives.prepare()
derivatives.run()
derivatives.analyse()
