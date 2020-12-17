import pyscf.qmmm
from pyscf import gto, scf
import numpy as np
from pyscf import lib
import inspect
from functools import reduce
from pyscf.scf import cphf
from pyscf import lib
from pyscf.prop.nmr import rhf as rhf_nmr

angstrom = 1 / 0.52917721067

def DeltaV(mol,dL):
    mol.set_rinv_orig_(mol.atom_coords()[0])
    dV=mol.intor('int1e_rinv')*dL[0]
    mol.set_rinv_orig_(mol.atom_coords()[1])
    dV+=mol.intor('int1e_rinv')*dL[1]
    return -dV.reshape((1,dV.shape[0],dV.shape[1]))


def alchemy_pol_deriv(polobj, with_cphf=True):
    mf = polobj._scf
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:,~occidx]

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = np.einsum('i,ix->x', charges, coords) / charges.sum()
    int_r=DeltaV(mol,[.001,-.001])    ########   .001 as finite difference intervall  
    h1 = lib.einsum('xpq,pi,qj->xij', int_r, mo_coeff.conj(), orbo) #going to molecular orbitals?
    s1 = np.zeros_like(h1)
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ)
    if with_cphf:
        mo1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1, polobj.max_cycle_cphf, polobj.conv_tol)[0]
    else:
        mo1 = rhf_nmr._solve_mo1_uncoupled(mo_energy, mo_occ, h1, s1)[0]
    return mo1

def alch_deriv(mf,atoms=[0,1]):
    """ dL=[.001,-.001]
    """
    mol=mf.mol
    plo=mf.Polarizability()
    pa=alchemy_pol_deriv(plo)
    nao=mol.nao
    nocc=mol.nelectron//2 #RHF
    U=np.zeros((nao,nao))
    U[:,:nocc]=pa[0,:,:nocc]
    U=U-U.T
    O=np.diag(mf.mo_occ)
    C=mf.mo_coeff
    dP=C@(U@O-O@U)@C.T 
    return(U,dP) 