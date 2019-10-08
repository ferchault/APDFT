#!/usr/bin/env python
import pyscf.gto
import pyscf.scf
import pyscf.dft
import numpy as np

class ElectronicEPN(object):
    def __init__(self, Z1, Z2, distance):
        """ Build and cache a dimer (Z1, Z2) at `distance` bond length, given in angstrom. """
        mol = pyscf.gto.Mole(atom='%d 0 0 0.; %d %f 0 0' % (Z1, Z2, distance), basis='6-31G', verbose=0)
        mol.build()
        calc = pyscf.scf.RHF(mol)
        self._etot = calc.kernel()
        self._dm = calc.make_rdm1()
        self._mol = mol
        
    def epn(self, pos):
        """ Electrostatic potential in a.u. at position `pos` given in bohr."""
        pos = np.array([pos,0,0])
        self._mol.set_rinv_orig_(pos)
        return np.matmul(self._dm, self._mol.intor("int1e_rinv")).trace()
    
    def epn_total(self, pos):
        """ Electrostatic potential from the nuclei in a.u. at position `pos` given in bohr."""
        pot = 0
        for site in range(self._mol.natm):
            ds = abs(self._mol.atom_coords()[site, 0] - pos)
            if ds > 1e-4:
                pot += self._mol.atom_charges()[site] / ds
            else:
                return np.nan
        return self.epn(pos) - pot
    
    def density(self, pos):
        """ Electron density in a.u. at position `pos` given in bohr."""
        pos = np.array([[pos,0,0]])
        ao_value = pyscf.dft.numint.eval_ao(self._mol, pos, deriv=0)
        rho = pyscf.dft.numint.eval_rho(self._mol, ao_value, self._dm, xctype="LDA")
        return rho[0]

    def total_energy_with_NN(self):
        """ Total energy including N-N interactions in a.u."""
        return self._etot
    
    def electron_nuclear_interaction(self):
        """ Interaction energy between electrons and nuclei in a.u."""
        return sum([self.epn(_[2]) for _ in self._mol.atom_coords()])
