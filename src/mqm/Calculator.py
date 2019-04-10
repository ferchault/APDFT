#!/usr/bin/env python

class Calculator(object):
	""" A concurrency-safe blocking interface for an external QM code."""
	def __init__(self):
		raise NotImplementedError()

	def evaluate(self, coordinates, nuclear_numbers, nuclear_charges, grid, basisset, method):
		raise NotImplementedError()

class HortonCalculator(object):
	self._methods = {
		'HF': HortonCalculator._setup_hf,
		'LDA': HortonCalculator._setup_lda,
		}

	def __init__(self):
		import horton
		horton.log.set_level(0)

	@staticmethod
	def _setup_hf(kin, er, na):
		return [UTwoIndexTerm(kin, 'kin'), UDirectTerm(er, 'hartree'), UExchangeTerm(er, 'x_hf'), UTwoIndexTerm(na, 'ne')]

	@staticmethod
	def _setup_lda(kin, er, na):
		return [UTwoIndexTerm(kin, 'kin'), UGridGroup(obasis, grid, [UBeckeHartree(lmax=8), ULibXCLDA('x'), ULibXCLDA('c_vwn')]), UTwoIndexTerm(na, 'ne')]

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

		terms = self._methods[method](kin, er, na)
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
		gridpoints = grid.get_points()
		rho_alpha = obasis.compute_grid_density_dm(dm_alpha, gridpoints)
		rho_beta = obasis.compute_grid_density_dm(dm_beta, gridpoint)s
		rho_full = rho_alpha + rho_beta
