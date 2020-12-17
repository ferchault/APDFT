import pyscf.qmmm
from pyscf import gto, scf
import numpy as np
from pyscf import lib
from functools import reduce
from pyscf.scf import cphf
from pyscf import lib
from pyscf.prop.nmr import rhf as rhf_nmr
dl=.001

angstrom = 1 / 0.52917721067

def DeltaV(mol,dL):
    mol.set_rinv_orig_(mol.atom_coords()[dL[0][0]])
    dV=mol.intor('int1e_rinv')*dL[1][0]*dl
    for i in range(1,len(dL[0])): 
        mol.set_rinv_orig_(mol.atom_coords()[dL[0][i]])
        dV+=mol.intor('int1e_rinv')*dL[1][i]*dl
    return -dV.reshape((1,dV.shape[0],dV.shape[1]))


def alchemy_pol_deriv(polobj,dL, with_cphf=True):
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
    
    int_r=DeltaV(mol,dL)    ########   .001 as finite difference intervall  
    h1 = lib.einsum('xpq,pi,qj->xij', int_r, mo_coeff.conj(), orbo) #going to molecular orbitals?
    s1 = np.zeros_like(h1)
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ)
    if with_cphf:
        mo1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1, polobj.max_cycle_cphf, polobj.conv_tol)[0]
    else:
        mo1 = rhf_nmr._solve_mo1_uncoupled(mo_energy, mo_occ, h1, s1)[0]
    return mo1

def alch_deriv(mf,dL=[0,1]):
    """ alch_deriv(mf,dL=[0,1]) returns U,dP for a dl=.001 times the charges
    dL can be the whole list of nuclear charges placed on atom, with length equals to mol.natm (eg.[0,1,0,0,-1,...,0])
    or alternatively a list with two sublist of equal length in the form [[atm_idxs],[atm_charges]]
    """
    mol=mf.mol
    a=[[],[]]
    parsed=False
    if len(dL) ==2:   #necessario, ma non sufficiente per caso 2
        try:
            len(dL[0])==len(dL[1])
            if isinstance(dL[0][0],int) or isinstance(dL[0][0],float):
                parsed=True
        except: pass
    elif mol.natm==len(dL) and (isinstance(dL[0],int) or isinstance(dL[0],float)): #move to case2 
        for i in range(len(dL)):
            if dL[i]!=0:
                a[0].append(i)
                a[1].append(dL[i])
        dL=a
        parsed=True

    if not parsed:
        print("Failed to parse charges")
        raise 
    print(dL)
    plo=mf.Polarizability()
    pa=alchemy_pol_deriv(plo,dL)
    nao=mol.nao
    nocc=mol.nelectron//2 #RHF
    U=np.zeros((nao,nao))
    U[:,:nocc]=pa[0,:,:nocc]
    U=U-U.T
    O=np.diag(mf.mo_occ)
    C=mf.mo_coeff
    dP=C@(U@O-O@U)@C.T 
    return(U,dP) 