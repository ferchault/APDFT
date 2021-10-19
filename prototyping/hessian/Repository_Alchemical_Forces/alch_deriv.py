import pyscf.qmmm
from pyscf import gto, scf
import numpy as np
import pyscf.qmmm
from pyscf import gto, scf
import numpy as np
from pyscf import lib
from functools import reduce
from pyscf.scf import cphf
from pyscf import lib
from pyscf.prop.nmr import rhf as rhf_nmr
from AP_utils import alias_param,parse_charge,DeltaV


def alchemy_cphf_deriv(mf,int_r, with_cphf=True):
    polobj=mf.Polarizability()
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
    h1 = lib.einsum('pq,pi,qj->ij', int_r, mo_coeff.conj(), orbo) #going to molecular orbitals
    h1=h1.reshape((1,h1.shape[0],h1.shape[1]))
    s1 = np.zeros_like(h1)
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ)
    if with_cphf:
        mo1,e1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1, polobj.max_cycle_cphf, polobj.conv_tol)
    else:
        mo1 = rhf_nmr._solve_mo1_uncoupled(mo_energy, mo_occ, h1, s1)[0]
    return mo1[0],e1[0]

def first_deriv_nuc_nuc(mol,dL):
    """dL=[[i1,i2,i3],[c1,c2,c3]]"""
    dnn=0
    for j in range(len(dL[0])):
        r2 = mol.atom_coord(dL[0][j]) 
        for i in range(mol.natm):
            if i != dL[0][j]:
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = np.linalg.norm(r1-r2)
                dnn += (q1 * dL[1][j])/ r
    return dnn

def second_deriv_nuc_nuc(mol,dL):
    """dL=[[i1,i2,i3],[c1,c2,c3]]"""
    dnn=0
    for j in range(len(dL[0])):
        r2 = mol.atom_coord(dL[0][j]) 
        for i in range(len(dL[0])):
            if dL[0][i] > dL[0][j]:
                r1 = mol.atom_coord(dL[0][i])
                r = np.linalg.norm(r1-r2)
                dnn += (dL[1][i] * dL[1][j])/ r
    return 2*dnn
                
def first_deriv_elec(mf,int_r):
    P=mf.make_rdm1()
    return np.einsum('ij,ji',P,int_r)

def second_deriv_elec(mf,int_r,mo1):
    orbo = mf.mo_coeff[:, :mo1.shape[1]]
    h1 = lib.einsum('pq,pi,qj->ij', int_r, mf.mo_coeff.conj(), orbo)
    e2 = np.einsum('pi,pi', h1, mo1)
    e2 *= 4
    return e2

def third_deriv_elec(mf,int_r,mo1,e1):   #only for one site (d^3 E /dZ^3)
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    mo1 = lib.einsum('qi,pq->pi', mo1, mo_coeff)
    dm1 = lib.einsum('pi,qi->pq', mo1, orbo) * 2
    dm1 = dm1 + dm1.transpose(1,0)
    vresp = mf.gen_response(hermi=1) # (J-K/2)(dm)
    h1ao = int_r + vresp(dm1)#Fock matrix
    e3  = lib.einsum('pq,pi,qi', h1ao, mo1, mo1) * 2  # *2 for double occupancy
    e3 -= lib.einsum('pq,pi,qj,ij', mf.get_ovlp(), mo1, mo1, e1) * 2
    e3 *=6
    return e3


def alch_deriv(mf,dL=[]):
    """ alch_deriv(mf,dL=[]) returns U,dP for a dl=.001 times the charges
    dL can be the whole list of nuclear charges placed on atom, with length equals to mol.natm (eg.[0,1,0,0,-1,...,0])
    or alternatively a list with two sublist of equal length in the form [[atm_idxs],[atm_charges]]
    """
    mol=mf.mol
    dL=parse_charge(dL)
    int_r=DeltaV(mol,dL)    
    mo1,e1=alchemy_cphf_deriv(mf,int_r)
    der1=first_deriv_elec(mf,int_r)+first_deriv_nuc_nuc(mol,dL)
    der2=second_deriv_elec(mf,int_r,mo1)+second_deriv_nuc_nuc(mol,dL)
    der3=third_deriv_elec(mf,int_r,mo1,e1)
    return (der1,der2,der3)

def make_dP(mf,mo1):
    mol=mf.mol
    nao=mol.nao
    nocc=mf.mol.nelec[0]
    C=mf.mo_coeff
    dP=np.zeros_like(C)
    dP[:,:]=2*np.einsum('ij,jk,lk->il',C,mo1,C[:,:nocc])
    return dP+dP.T

def make_U(mo1):
    U=np.zeros((mo1.shape[0],mo1.shape[0]))
    U[:,:mo1.shape[1]]=mo1
    U=U-U.T
    return U


def alch_hessian(mf,int_r,mo1):
    mo_coeff=mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]    
    h1 = lib.einsum('xpq,pi,qj->xij', int_r, mo_coeff.conj(), orbo)
    e2 = np.einsum('xpi,ypi->xy', h1, mo1)
    e2 = (e2 + e2.T) * 2
    return e2
    
def cubic_alch_hessian(mf,int_r,mo1,e1):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    mo1 = lib.einsum('xqi,pq->xpi', mo1, mo_coeff)           #dC=UC
    dm1 = lib.einsum('xpi,qi->xpq', mo1, orbo) * 2
    dm1 = dm1 + dm1.transpose(0,2,1)            #dP= dCOC^T+COdC'T
    vresp = mf.gen_response(hermi=1)
    h1ao = int_r + vresp(dm1)         # dF=dV+G(dP)
    # *2 for double occupancy
    e3  = lib.einsum('xpq,ypi,zqi->xyz', h1ao, mo1, mo1) * 2   # trace( dC^T dF dC)
    e3 -= lib.einsum('pq,xpi,yqj,zij->xyz', mf.get_ovlp(), mo1, mo1, e1) * 2  # - dC^T S dC de
    e3 = (e3 + e3.transpose(1,2,0) + e3.transpose(2,0,1) +
          e3.transpose(0,2,1) + e3.transpose(1,0,2) + e3.transpose(2,1,0))
    return e3