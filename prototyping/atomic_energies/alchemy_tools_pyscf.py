import numpy as np
import scipy as sc
import numpy
import pyscf
import pyscf.gto
import pyscf.qmmm
import pyscf.scf
import pyscf.dft
import pyscf.lib
from pyscf.data import nist
import qml
from ase import units

def add_qmmm(calc, deltaZ, includeonly, mol):
    """
    modify hamiltonian such that Z = Z(lambda)
    """
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords()[includeonly]*units.Bohr, deltaZ) # add charge dZ at position of nuclei

    def energy_nuc(self):
        """
        calculate correct nuclear charge for modified system
        """
        q = mol.atom_charges().astype(np.float)
        q[includeonly] += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf

def calculate_alchpot(dm1_ao, includeonly, mol):
    # Electronic EPN from electron density
    alch_pots = []
    for site in includeonly:
        mol.set_rinv_orig_(mol.atom_coords()[site])
        alch_pots.append(np.matmul(dm1_ao, mol.intor("int1e_rinv")).trace())
    return(-np.array(alch_pots))

def calculate_average_alchpots(alchpots, lam_vals, intg_meth, even=None):
    num_alchpots = len(alchpots[0])
    av_alchpots = []
    for i in range(num_alchpots):
        if intg_meth == 'trapz':
            av_alchpot_i = np.trapz(alchpots[:,i], lam_vals)
        elif intg_meth == 'simps':
#             assert even != None or len(lam_vals)%2 != 0, 'How to handle even number of samples?'
            av_alchpot_i = sc.integrate.simps(alchpots[:,i], lam_vals, even=even)
        av_alchpots.append(av_alchpot_i)
    return(np.array(av_alchpots))

def get_num_elec(lam_val, total_num_elecs):
    """
    calculate number of electrons for a given lambda value and total number of electrons
    """
    if int(lam_val*total_num_elecs)%2 == 0:
        num_elec = int(lam_val*total_num_elecs)
    else:
        num_elec = int(lam_val*total_num_elecs) + 1
    return(num_elec)

def make_apdft_calc(deltaZ, dm_restart, includeonly, mol, method = "HF", **kwargs):
    """
    SCF calculation for fractional charges defined in deltaZ
    returns the density matrix and the total energy
    """
    
    if 'verbose' in kwargs.keys():
        verbose = kwargs['verbose']
    else:
        verbose = 0
    
    if method not in ["CCSD", "HF"]:
        raise NotImplementedError("Method %s not supported." % method)
    
    
    if method == "HF":
        calc = add_qmmm(pyscf.scf.RHF(mol), deltaZ, includeonly, mol)
        for k in kwargs.keys():
            if k == 'max_cycle':
                calc.max_cycle = kwargs[k]
            elif k == 'diis':
                calc.diis = kwargs[k]
            elif k == 'init_guess':
                calc.init_guess = kwargs[k]
        if dm_restart is None:
            hfe = calc.kernel(verbose=verbose)
        else:
            hfe = calc.kernel(dm0 = dm_restart, verbose=verbose)
        dm1_ao = calc.make_rdm1()
        total_energy = calc.e_tot
#     if method == "CCSD":
#         calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
#         hfe = calc.kernel(verbose=verbose)
#         mycc = pyscf.cc.CCSD(calc).run()
#         dm1 = mycc.make_rdm1()
#         dm1_ao = np.einsum("pi,ij,qj->pq", calc.mo_coeff, dm1, calc.mo_coeff.conj())
#         total_energy = mycc.e_tot
    return(dm1_ao, total_energy, calc.mo_energy, calc.mo_occ)

def prepare_input(coords, nuc_charges, num_elec, basis = 'def2-qzvp'):
    lam_val = num_elec/nuc_charges.sum()
    
    mol = pyscf.gto.Mole()
    for ch, coords_atom in zip(nuc_charges, coords):
        mol.atom.append([ch, coords_atom])
    mol.basis = basis
    mol.charge = nuc_charges.sum() - num_elec
    mol.build()
    # dZ vector to generate systems for lambda != 1
    deltaZ = -nuc_charges*(1-lam_val)
    includeonly = np.arange(len(mol.atom_coords()))
    
    return(deltaZ, includeonly, mol)
    
