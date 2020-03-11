#!/usr/bin/env python
import pyscf.gto
import pyscf.scf
import pyscf.dft
import numpy as np
import quadpy
import matplotlib.pyplot as plt

class Calculator(object):
    def __init__(self, zs, poss, spin):
        atomspec = ['%d %f 0 0' % _ for _ in zip(zs, poss)]
        mol = pyscf.gto.Mole(atom=';'.join(atomspec), basis='6-31G', verbose=0)
        mol.spin = spin
        mol.build()
        calc = pyscf.scf.UHF(mol)
        self._etot = calc.kernel()
        self._dm = calc.make_rdm1()
        self._mol = mol
        self._grid_scheme = quadpy.sphere.lebedev_077()
        self._grid_ds = np.linspace(0.01, 20, 300)
    
    def epn(self, pos):
        """ Electrostatic potential from the electrons in a.u. at position `pos` given in bohr."""
        pos = np.array(pos)
        self._mol.set_rinv_orig_(pos)
        alpha = np.matmul(self._dm[0], self._mol.intor("int1e_rinv")).trace()
        beta = np.matmul(self._dm[1], self._mol.intor("int1e_rinv")).trace()
        return alpha+beta
    
    def epn_total(self, pos):
        """ Electrostatic potential from the nuclei in a.u. at position `pos` given in bohr."""
        pot = 0
        for site in range(self._mol.natm):
            ds = np.linalg.norm(self._mol.atom_coords()[site] - pos)
            if ds > 1e-4:
                pot += self._mol.atom_charges()[site] / ds
            else:
                return np.nan
        return self.epn(pos) - pot
    
    def density(self, pos):
        """ Electron density in a.u. at position `pos` given in bohr."""
        pos = np.array(pos)
        assert len(pos.shape) == 2, 'Coordinates need to be Nx3 array.'
        assert pos.shape[1] == 3, 'Coordinates need to be in in 3 dim.'
        ao_value = pyscf.dft.numint.eval_ao(self._mol, pos, deriv=0)
        rho_alpha = pyscf.dft.numint.eval_rho(self._mol, ao_value, self._dm[0], xctype="LDA")
        rho_beta = pyscf.dft.numint.eval_rho(self._mol, ao_value, self._dm[1], xctype="LDA")
        return rho_alpha + rho_beta

    def total_energy_with_NN(self):
        """ Total energy including N-N interactions in a.u."""
        return self._etot
    
    def electron_nuclear_interaction(self):
        """ Interaction energy between electrons and nuclei in a.u."""
        return sum([self.epn(_) for _ in self._mol.atom_coords()])
    
    def density3d(self, coordinates):
        ao_value = pyscf.dft.numint.eval_ao(self._mol, coordinates, deriv=0)
        
        rho_alpha = pyscf.dft.numint.eval_rho(self._mol, ao_value, self._dm[0], xctype="LDA")
        rho_beta = pyscf.dft.numint.eval_rho(self._mol, ao_value, self._dm[1], xctype="LDA")
        return rho_alpha + rho_beta

    def radial_epn(self, pos):        
        epns = []
        dr = self._grid_ds[1] - self._grid_ds[0]
        for d in self._grid_ds:
            pts = self._grid_scheme.points * d + np.array(pos)
            rho = self.density3d(pts)
            epn = sum(rho * self._grid_scheme.weights / d)
            epn *= 4*np.pi * d**2 * dr
            epns.append(epn)
        
        return np.array(epns)

    def get_adaptive_grid(self):
        grid = pyscf.dft.gen_grid.Grids(self._mol)
        grid.level = 3
        grid.build()
        return grid.coords
    
class ElectronicEPN(object):
    def __init__(self, z1, z2, distance, mode=None):
        """ Build and cache a dimer (Z1, Z2) at `distance` bond length, given in angstrom. """
        self._calculator = Calculator((z1, z2), (0, distance), (z1 + z2) % 2)
        self._mode = mode
        
        spins = [0, 1, 0, 1, 0, 1, 2, 3, 2, 1, 0]
        if self._mode == 'remove_free_atom_density':
            self._atom1 = Calculator((z1,), (0,), spins[z1])
            self._atom2 = Calculator((z2,), (distance,), spins[z2])
        
    def epn_total(self, pos):
        """ Total EPN (including nuclei) at pos."""
        if self._mode == 'remove_free_atom_density':
            return self._calculator.epn_total(pos) - self._atom1.epn_total(pos) - self._atom2.epn_total(pos)
        return self._calculator.epn_total(pos)
    
    def epn(self, pos):
        """ The true electrostatic potential at `pos`."""
        if self._mode == 'remove_free_atom_density':
            return self._calculator.epn(pos) - self._atom1.epn(pos) - self._atom2.epn(pos)
        return self._calculator.epn(pos)
    
    def epd(self, pos):
        """ The electronic potential of the free atoms radially resolved."""
        if self._mode != 'remove_free_atom_density':
            raise NotImplementedError()
        
        return self._atom1.radial_epn(pos) + self._atom2.radial_epn(pos)

    def density(self, pos):
        """ The electron density at pos along the bond axis."""
        if self._mode == 'remove_free_atom_density':
            return self._calculator.density(pos) - self._atom1.density(pos) - self._atom2.density(pos)
        return self._calculator.density(pos)
    
    def total_energy_with_NN(self):
        return self._calculator.total_energy_with_NN()

    def electron_nuclear_interaction(self):
        return self._calculator.electron_nuclear_interaction()

    def get_adaptive_grid(self):
        return self._calculator.get_adaptive_grid()

if __name__ == '__main__':
    e = ElectronicEPN(6, 8, 2, 'remove_free_atom_density')
    print (e.density([[0, 1, 2], [1, 2, 3]])) 
    print (e.density([[0, 1, 2]])) 
    print (e.get_adaptive_grid().shape)
