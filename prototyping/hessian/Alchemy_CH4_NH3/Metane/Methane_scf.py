from horton import *
import numpy as np
def UHF(mol,basis_set='sto-3g'):

    obasis=get_gobasis(mol.coordinates,mol.numbers,basis_set)
    lf=DenseLinalgFactory(obasis.nbasis)
    #orbital integrals
    olp=obasis.compute_overlap(lf)
    kin=obasis.compute_kinetic(lf)
    na=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
    er=obasis.compute_electron_repulsion(lf)

    exp_alpha=lf.create_expansion()
    exp_beta=lf.create_expansion()
    
    guess_core_hamiltonian(olp, kin, na, exp_alpha,exp_beta)
    #exp_alpha.randomize()
    #exp_beta.randomize()
    
    occ_model=AufbauOccModel(5,5)
    occ_model.assign(exp_alpha, exp_beta)
    
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        UTwoIndexTerm(kin, 'kin'),
        UDirectTerm(er, 'hartree'),
        UExchangeTerm(er, 'x_hf'),
        UTwoIndexTerm(na, 'ne'),
    ]
    ham = UEffHam(terms, external)
    # Converge WFN with Optimal damping algorithm (ODA) SCF
    # - Construct the initial density matrix (needed for ODA).
    
    dm_alpha = exp_alpha.to_dm()
    dm_beta = exp_beta.to_dm()
    # - SCF solver
    scf_solver = EDIIS2SCFSolver(1e-9,maxiter=300)
    scf_solver(ham, lf, olp, occ_model, dm_alpha, dm_beta)
    return(ham.cache['energy'])


    cart_array= np.asarray(cart_geom)

