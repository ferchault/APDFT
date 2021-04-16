from pyscf import scf,gto
import numpy as np
from pyscf.grad import rhf as grhf
from alch_deriv import alch_deriv

NUC_FRAC_CHARGE=gto.mole.NUC_FRAC_CHARGE
NUC_MOD_OF=gto.mole.NUC_MOD_OF
PTR_FRAC_CHARGE=gto.mole.PTR_FRAC_CHARGE


def DeltaV(mol,dL):
    mol.set_rinv_orig_(mol.atom_coords()[0])
    dV=mol.intor('int1e_rinv')*dL[0]
    if len(dL)>1:
        for i in range(1,len(dL)):
            mol.set_rinv_orig_(mol.atom_coords()[i])
            dV+=mol.intor('int1e_rinv')*dL[i]
    return -dV



#molecola /FCM + dP/dZ -> d^2 E_e/ dZ dX solo elettronic gradient 
 

def g1(mol0,dP,P,DZ,g0):    # dP/dz*dH/dx, P* d2H/dzdx 
    natm=mol0.natm
    nao=mol0.nao
    denv=mol0._env.copy()
    datm=mol0._atm.copy()
    datm[:,NUC_MOD_OF] = NUC_FRAC_CHARGE
    for i in range(natm):
        denv[datm[i,PTR_FRAC_CHARGE]]=DZ[i]
    dH1=-gto.moleintor.getints('int1e_ipnuc_sph',datm,mol0._bas,denv, None,3,0,'s1')
    dH_dxdz=np.zeros((natm,3,nao,nao))
    for atm_id in range(natm):
        with mol0.with_rinv_at_nucleus(atm_id):
            vrinv = -mol0.intor('int1e_iprinv', comp=3)
        shl0, shl1, p0, p1 = mol0.aoslice_by_atom()[atm_id]
        vrinv*=DZ[atm_id]
        vrinv[:,p0:p1] += dH1[:,p0:p1]  
        vrinv += vrinv.transpose(0,2,1)
        dH_dxdz[atm_id]=vrinv
    ga_1=np.zeros((natm,3))    
    for i in range(natm):
        ga_1[i]+=np.einsum('xij,ij->x', g0.hcore_generator()(i),dP)
        ga_1[i]+=np.einsum('xij,ij->x', dH_dxdz[i],P)
    return(ga_1)

def g2(mol0,dP,P,DZ,g0):
    natm=mol0.natm
    nao=mol0.nao
    aoslices = mol0.aoslice_by_atom()
    ga_2=np.zeros((natm,3))
    vhf = g0.get_veff(mol0, P)
    vhf_1 = g0.get_veff(mol0, P+dP)
    for ia in range(natm):
        p0, p1 = aoslices [ia,2:]
        ga_2[ia]=(np.einsum('xij,ij->x', vhf[:,p0:p1], dP[p0:p1]) * 2)
        ga_2[ia]+=(np.einsum('xij,ij->x',vhf_1[:,p0:p1]-vhf[:,p0:p1], P[p0:p1]) * 2)
    return(ga_2)
    
def g3(mol0,dP,P,g0,e,e1,C,dC): #-dW/dZ *dS/dx
    s1=g0.get_ovlp(mol0)
    g3=np.zeros((mol0.natm,3))
    nocc=mol0.nelec[0]
    dW=np.einsum('i,ji,ki->jk' ,2*e[:nocc],dC[:,:nocc],C[:,:nocc])+np.einsum('i,ji,ki->jk', 2*e[:nocc],C[:,:nocc],dC[:,:nocc])+  2*C[:,:nocc]@e1@C.T[:nocc,:]
    for i in range(mol0.natm):
        p0, p1 = mol0.aoslice_by_atom() [i,2:]
        g3[i] -= np.einsum('xij,ij->x', s1[:,p0:p1], dW[p0:p1]) * 2
    return(g3)
    
def aaff(mf,DZ):
    mol0=mf.mol
    g0=mf.Gradients()
    P=mf.make_rdm1()
    C=mf.mo_coeff
    e=mf.mo_energy
    U,dP,e1=alch_deriv(mf,[[x for x in range(mol0.natm)],DZ])
    dC=C@U
    return g1(mol0,dP,P,DZ,g0)+g2(mol0,dP,P,DZ,g0)+g3(mol0,dP,P,g0,e,e1,C,dC)


def alc_deriv_grad_nuc(mol,dL):  # to get the derivative with respect to alch. perturbation
    gn = np.zeros((mol.natm,3))
    for j in range(mol.natm):
        q2 =  mol.atom_charge(j)
        r2 = mol.atom_coord(j) 
        for i in range(mol.natm):
            if i != j:
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = np.linalg.norm(r1-r2)
                gn[j] += (q1 * dL[j]+ q2*dL[i]) * (r1-r2) / r**3
    return gn

def alc_differential_grad_nuc(mol,dL):  # to get the exact diffeential after alch. perturbation
    gn = np.zeros((mol.natm,3))
    for j in range(mol.natm):
        q2 =  mol.atom_charge(j) + dL[j]
        r2 = mol.atom_coord(j) 
        for i in range(mol.natm):
            if i != j:
                q1 = mol.atom_charge(i) +dL[i]
                r1 = mol.atom_coord(i)
                r = np.linalg.norm(r1-r2)
                gn[j] += (q1 * q2-mol.atom_charge(i)*mol.atom_charge(j)) * (r1-r2) / r**3
    return gn 


def g3_old(mol0,dP,P,DZ,g0,vresp,F): #-dW/dZ *dS/dx
    s1=g0.get_ovlp(mol0)
    dF=vresp(dP)+DeltaV(mol0,DZ)
    S=mol0.intor_symmetric('int1e_ovlp')
    S_1=np.linalg.inv(S)
    g3=np.zeros((mol0.natm,3))
    for i in range(mol0.natm):
        p0, p1 = mol0.aoslice_by_atom() [i,2:]
        g3[i] -= np.einsum('xij,ij->x', s1[:,p0:p1], (S_1@((F@dP)+(dF@P)))[p0:p1]) * 2
    return(g3)    
