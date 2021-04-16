import pyscf.qmmm
from pyscf import gto, scf
import numpy as np
from pyscf import lib
from functools import reduce
from pyscf.scf import cphf
from pyscf import lib
from pyscf.prop.nmr import rhf as rhf_nmr
angstrom = 1 / 0.52917721067

def DeltaV(mol,dL):
    mol.set_rinv_orig_(mol.atom_coords()[dL[0][0]])
    dV=mol.intor('int1e_rinv')*dL[1][0]
    for i in range(1,len(dL[0])): 
        mol.set_rinv_orig_(mol.atom_coords()[dL[0][i]])
        dV+=mol.intor('int1e_rinv')*dL[1][i]
    return -dV.reshape((1,dV.shape[0],dV.shape[1]))


def alchemy_pol_deriv(polobj,int_r, with_cphf=True):
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
    h1 = lib.einsum('xpq,pi,qj->xij', int_r, mo_coeff.conj(), orbo) #going to molecular orbitals?
    s1 = np.zeros_like(h1)
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ)
    if with_cphf:
        mo1,e1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1, polobj.max_cycle_cphf, polobj.conv_tol)
    else:
        mo1 = rhf_nmr._solve_mo1_uncoupled(mo_energy, mo_occ, h1, s1)[0]
    return mo1,e1

def third_deriv(polobj,mf,int_r,mo1,e1):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    
    mo1 = lib.einsum('xqi,pq->xpi', mo1, mo_coeff)
    dm1 = lib.einsum('xpi,qi->xpq', mo1, orbo) * 2
    dm1 = dm1 + dm1.transpose(0,2,1)
    vresp = mf.gen_response(hermi=1) # (J-K/2)(dm)
    
    h1ao = int_r + vresp(dm1) #Fock matrix
    e3  = lib.einsum('xpq,ypi,zqi->xyz', h1ao, mo1, mo1) * 2  # *2 for double occupancy
    e3 -= lib.einsum('pq,xpi,yqj,zij->xyz', mf.get_ovlp(), mo1, mo1, e1) * 2
    e3 = (e3 + e3.transpose(1,2,0) + e3.transpose(2,0,1) + e3.transpose(0,2,1) + e3.transpose(1,0,2) + e3.transpose(2,1,0))
   # e3 = -e3   CHECK THIS SIGN
    return e3
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
    polobj=mf.Polarizability()
    int_r=DeltaV(mol,dL)    
    mo1,e1=alchemy_pol_deriv(polobj,int_r)
    nao=mol.nao
    nocc=mol.nelectron//2 #RHF
    U=np.zeros((nao,nao))
    U[:,:nocc]=mo1[0,:,:nocc]
    U=U-U.T
    O=np.diag(mf.mo_occ)
    C=mf.mo_coeff
    dP=C@(U@O-O@U)@C.T    
    P=mf.make_rdm1()
    der1=np.einsum('ij,ij',P,int_r[0])
    der2=np.einsum('ij,ij', int_r[0,:,:nocc], mo1[0])
    #der2=np.einsum('xpi,ypi->xy', int_r, mo1)
    der3=third_deriv(polobj,mf,int_r,mo1,e1)[0,0,0]
    return (der1,der2,der3)
    #return(U,dP,e1) 