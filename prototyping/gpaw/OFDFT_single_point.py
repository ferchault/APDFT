#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:30:06 2019

@author: misa
"""

from ase import Atoms
from ase.parallel import paropen
from gpaw import GPAW
from gpaw.mixer import Mixer
from gpaw.eigensolvers import CG
from gpaw.poisson import PoissonSolver
from gpaw import setup_paths
from ase.units import Bohr, Hartree
setup_paths.insert(0, '.')

# Usual GPAW definitions
h = 0.18
a = 12.00
c = a/2

# XC functional + kinetic functional (minus the Tw contribution) to be used

#xcname = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'

#xcname='1.0_LDA_K_TF+1.0_LDA_K_LP+1.0_LDA_X+1.0_LDA_C_PW'

#xcname='1.0_LDA_K_TF+1.0_LDA_K_LP+-1.0_LDA_K_LP+1.0_LDA_X+1.0_LDA_C_PW'

xcname='1.0_GGA_K_TFVW+-1.0_GGA_K_VW+1.0_GGA_X_PBE+1.0_GGA_C_PBE'

# Fraction of Tw
lambda_coeff = 1.0

name = 'lambda_{0}'.format(lambda_coeff)

filename = 'atoms_'+name+'.dat'

f = paropen(filename, 'w')

elements = 'N'


mixer = Mixer()

eigensolver = CG(tw_coeff=lambda_coeff)

poissonsolver=PoissonSolver()
molecule = Atoms(elements,
                 positions=[(c,c,c)] ,
                 cell=(a,a,a))

calc = GPAW(h=h,
            xc=xcname,
            maxiter=240,
            eigensolver=eigensolver,
            mixer=mixer,
            setups=name,
            poissonsolver=poissonsolver)

molecule.set_calculator(calc)

E = molecule.get_total_energy()

f.write('{0}\t{1}\n'.format(elements,E))