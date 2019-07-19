#!/usr/bin/env python
import horton

method = "{{ method }}"
mol = horton.IOData()
mol.coordinates = np.array({{coordinates}})
mol.numbers = np.array({{nuclear_numbers}})
mol.pseudo_numbers = np.array({{nuclear_charges}})

obasis = horton.get_gobasis(mol.coordinates, nuclear_numbers, "{{ basisset }}")

olp = obasis.compute_overlap()
kin = obasis.compute_kinetic()
na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
er = obasis.compute_electron_repulsion()

orb_alpha = horton.Orbitals(obasis.nbasis)
orb_beta = horton.Orbitals(obasis.nbasis)

one = kin + na
horton.guess_core_hamiltonian(olp, one, orb_alpha, orb_beta)

external = {"nn": horton.compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}

grid = horton.BeckeMolGrid(
    mol.coordinates,
    mol.numbers,
    mol.pseudo_numbers,
    "fine",
    mode="keep",
    random_rotate=False,
)

if method == "HF":
    terms = [
        horton.UTwoIndexTerm(kin, "kin"),
        horton.UDirectTerm(er, "hartree"),
        horton.UExchangeTerm(er, "x_hf"),
        horton.UTwoIndexTerm(na, "ne"),
    ]
if method == "LDA":
    terms = [
        horton.UTwoIndexTerm(kin, "kin"),
        horton.UGridGroup(
            obasis,
            grid,
            [
                horton.UBeckeHartree(lmax=8),
                horton.ULibXCLDA("x"),
                horton.ULibXCLDA("c_vwn"),
            ],
        ),
        horton.UTwoIndexTerm(na, "ne"),
    ]
if method == "PBE":
    terms = [
        horton.UTwoIndexTerm(kin, "kin"),
        horton.UDirectTerm(er, "hartree"),
        horton.UGridGroup(
            obasis, grid, [horton.ULibXCGGA("x_pbe"), horton.ULibXCGGA("c_pbe")]
        ),
        horton.UTwoIndexTerm(na, "ne"),
    ]
if method == "PBE0":
    libxc_term = horton.ULibXCHybridGGA("xc_pbe0_13")
    terms = [
        horton.UTwoIndexTerm(kin, "kin"),
        horton.UDirectTerm(er, "hartree"),
        horton.UGridGroup(obasis, grid, [libxc_term]),
        horton.UExchangeTerm(er, "x_hf", libxc_term.get_exx_fraction()),
        horton.UTwoIndexTerm(na, "ne"),
    ]

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
