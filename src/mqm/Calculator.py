#!/usr/bin/env python
import numpy as np

class Calculator(object):
	""" A concurrency-safe blocking interface for an external QM code."""
	def __init__(self):
		raise NotImplementedError()

	def evaluate(self, coordinates, nuclear_numbers, nuclear_charges, grid, basisset, method):
		raise NotImplementedError()


def _horton_setup_hf(obasis, grid, kin, er, na):
	return [horton.UTwoIndexTerm(kin, 'kin'), horton.UDirectTerm(er, 'hartree'), horton.UExchangeTerm(er, 'x_hf'), horton.UTwoIndexTerm(na, 'ne')]

def _horton_setup_lda(obasis, grid, kin, er, na):
	return [horton.UTwoIndexTerm(kin, 'kin'), horton.UGridGroup(obasis, grid, [horton.UBeckeHartree(lmax=8), horton.ULibXCLDA('x'), horton.ULibXCLDA('c_vwn')]), horton.UTwoIndexTerm(na, 'ne')]

def _horton_setup_pbe(obasis, grid, kin, er, na):
	return [horton.UTwoIndexTerm(kin, 'kin'), horton.UDirectTerm(er, 'hartree'), horton.UGridGroup(obasis, grid, [horton.ULibXCGGA('x_pbe'), horton.ULibXCGGA('c_pbe')]), horton.UTwoIndexTerm(na, 'ne')]

def _horton_setup_pbe0(obasis, grid, kin, er, na):
	libxc_term = horton.ULibXCHybridGGA('xc_pbe0_13')
	return [horton.UTwoIndexTerm(kin, 'kin'), horton.UDirectTerm(er, 'hartree'), horton.UGridGroup(obasis, grid, [libxc_term]), horton.UExchangeTerm(er, 'x_hf', libxc_term.get_exx_fraction()), horton.UTwoIndexTerm(na, 'ne')]

class HortonCalculator(Calculator):
	_methods = {
		'HF': _horton_setup_hf,
		'LDA': _horton_setup_lda,
		'PBE': _horton_setup_pbe,
		'PBE0': _horton_setup_pbe0,
		}

	def __init__(self):
		# lazy import only if available
		global horton
		import horton
		horton.log.set_level(0)

	def evaluate(self, coordinates, nuclear_numbers, nuclear_charges, grid, basisset, method):
		mol = horton.IOData()
		mol.coordinates = coordinates
		mol.numbers = nuclear_numbers
		mol.pseudo_numbers = nuclear_charges

		obasis = horton.get_gobasis(mol.coordinates, nuclear_numbers, basisset)

		olp = obasis.compute_overlap()
		kin = obasis.compute_kinetic()
		na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
		er = obasis.compute_electron_repulsion()

		orb_alpha = horton.Orbitals(obasis.nbasis)
		orb_beta = horton.Orbitals(obasis.nbasis)

		one = kin + na
		horton.guess_core_hamiltonian(olp, one, orb_alpha, orb_beta)

		external = {'nn': horton.compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}

		grid = horton.BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, 'fine', mode='keep', random_rotate=False)

		terms = self._methods[method](obasis, grid, kin, er, na)
		ham = horton.UEffHam(terms, external)
		occ_model = horton.AufbauOccModel(1, 1)
		occ_model.assign(orb_alpha, orb_beta)
		dm_alpha = orb_alpha.to_dm()
		dm_beta = orb_beta.to_dm()
		scf_solver = horton.EDIIS2SCFSolver(1e-5, maxiter=400)
		scf_solver(ham, olp, occ_model, dm_alpha, dm_beta)

		fock_alpha = np.zeros(olp.shape)
		fock_beta = np.zeros(olp.shape)
		ham.reset(dm_alpha, dm_beta)
		energy = ham.compute_energy()
		ham.compute_fock(fock_alpha, fock_beta)
		orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)
		orb_beta.from_fock_and_dm(fock_beta, dm_beta, olp)

		# integration grid
		rho_alpha = obasis.compute_grid_density_dm(dm_alpha, grid.points)
		rho_beta = obasis.compute_grid_density_dm(dm_beta, grid.points)
		rho_full = rho_alpha + rho_beta
