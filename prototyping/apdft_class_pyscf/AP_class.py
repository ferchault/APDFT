# A class for cphf apdft perturbation up to 3rd order
import pyscf.qmmm
from pyscf import gto, scf
import numpy as np
from pyscf import lib
from functools import reduce
from pyscf.scf import cphf
from pyscf import lib
from pyscf.prop.nmr import rhf as rhf_nmr
#from aaff import alc_deriv_grad_nuc,aaff
from scipy.spatial.transform import Rotation as R
from FcMole import FcM_like

ang2bohr=1.8897261246
bohr2ang=.5291772109
charge2symbol={1:"H",2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne"}

def DeltaV(mol,dL):
    """dL=[[i1,i2,i3],[c1,c2,c3]]"""
    mol.set_rinv_orig_(mol.atom_coords()[dL[0][0]])
    dV=mol.intor('int1e_rinv')*dL[1][0]
    for i in range(1,len(dL[0])): 
        mol.set_rinv_orig_(mol.atom_coords()[dL[0][i]])
        dV+=mol.intor('int1e_rinv')*dL[1][i]
    return -dV


def parse_charge(dL):
    a=[[],[]]
    parsed=False
    if len(dL) ==2:   
        try:
            len(dL[0])==len(dL[1])
            if isinstance(dL[0][0],int) or isinstance(dL[0][0],float):
                parsed=True
        except: pass
    if not parsed and (isinstance(dL[0],int) or isinstance(dL[0],float)): #move to case2 
        for i in range(len(dL)):
            if dL[i]!=0:
                a[0].append(i)
                a[1].append(dL[i])
        dL=a
        parsed=True
    if not parsed:
        print("Failed to parse charges")
        raise
    return dL

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

def third_deriv_elec(mf,int_r,mo1,e1):
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


#symmetry operations on bezene
def old_rotate_matrix(M,mol,atm_idx):  # to rotate the idx of the carbon atoms
    pt=mol.aoslice_by_atom()[atm_idx,-2]
    Mr=np.zeros_like(M)
    Mr[:-pt,:-pt]=M[pt:,pt:]
    Mr[-pt:,-pt:]=M[:pt,:pt]
    Mr[:-pt,-pt:]=M[pt:,:pt]
    Mr[-pt:,:-pt]=M[:pt,pt:]
    return Mr

def rotate_matrix(M,mol,atm_idx,ref_site=0):  # to rotate the idx of the carbon atoms
    pt=mol.aoslice_by_atom()[atm_idx,-2]
    rpt=mol.aoslice_by_atom()[ref_site,-2]
    Mr=np.zeros_like(M)
    msize=M.shape[0]
    for i in range(msize):
            for j in range(msize):
                Mr[i,j]=M[(i+rpt-pt)%msize,(j+rpt-pt)%msize]
    return Mr

def rotate_grad(g,atm_idx,ref_site=0):
    gr=np.zeros_like(g)
    glen=g.shape[0]
    for i in range(glen):
        gr[i]=g[(i+ref_site-atm_idx)%glen]
    return gr

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
    mo1 = lib.einsum('xqi,pq->xpi', mo1, mo_coeff)
    dm1 = lib.einsum('xpi,qi->xpq', mo1, orbo) * 2
    dm1 = dm1 + dm1.transpose(0,2,1)
    vresp = mf.gen_response(hermi=1)
    h1ao = int_r + vresp(dm1)
    # *2 for double occupancy
    e3  = lib.einsum('xpq,ypi,zqi->xyz', h1ao, mo1, mo1) * 2
    e3 -= lib.einsum('pq,xpi,yqj,zij->xyz', mf.get_ovlp(), mo1, mo1, e1) * 2
    e3 = (e3 + e3.transpose(1,2,0) + e3.transpose(2,0,1) +
          e3.transpose(0,2,1) + e3.transpose(1,0,2) + e3.transpose(2,1,0))
    e3 = -e3
    return e3

class APDFT_perturbator(lib.StreamObject):
    def __init__(self,mf,symmetry=None,sites=None):
        self.mf=mf
        self.mol=mf.mol
        self.symmetry=symmetry
        self.sites=[]
        for site in sites: self.sites.append(site)
        self.DeltaV=DeltaV
        self.alchemy_cphf_deriv=alchemy_cphf_deriv
        self.make_dP=make_dP
        self.make_U=make_dP
        self.dVs={}
        self.mo1s={}
        self.e1s={}
        self.dPs={}
        #self.afs={}
        self.perturb()
        #self.symm=benz_Symm(self.mol)
        self.cubic_hessian=None
        self.hessian=None
        self.gradient=None
    
    def U(self,atm_idx):
        if atm_idx not in self.sites:
                self.sites.append(atm_idx)
                self.perturb()
        return make_U(self.mo1s[atm_idx])
    def dP(self,atm_idx):
        if atm_idx not in self.sites:
                self.sites.append(atm_idx)
                self.perturb()
        return make_dP(self.mf,self.mo1s[atm_idx])
    
    def perturb(self):
        for site in self.sites:
            if not site in self.mo1s:
                self.dVs[site]=DeltaV(self.mol,[[site],[1]])
                self.mo1s[site],self.e1s[site]=alchemy_cphf_deriv(self.mf,self.dVs[site])
    def mo1(self,atm_idx):
        if atm_idx not in self.mo1s:
            if atm_idx not in self.sites:
                self.sites.append(atm_idx)
            self.perturb()
        return self.mo1s[atm_idx]
        
    def dV(self,atm_idx):
        if atm_idx not in self.dVs: 
            if atm_idx not in self.sites:
                self.sites.append(atm_idx)
            self.perturb()
        return self.dVs[atm_idx]
        
    def e1(self,atm_idx):
        if atm_idx not in self.e1s: 
            if atm_idx not in self.sites:
                self.sites.append(atm_idx)
            self.perturb()
        return self.e1s[atm_idx]
        
#    def af(self,atm_idx):
#        if atm_idx in self.afs: 
#            return self.afs[atm_idx] 
#        elif atm_idx in self.symm.eqs:
#            afr=self.af(self.symm.eqs[atm_idx]['ref'])
#            self.afs[atm_idx]=rotate_grad(self.symm.eqs[atm_idx]['op'].apply(afr),atm_idx,ref_site=self.symm.eqs[atm_idx]['ref'])
#        else:
#            print("No AF found for atom {}. Calculating it from code".format(atm_idx))
#            if atm_idx not in self.sites:
#                self.sites.append(atm_idx)
#                self.perturb()
#            DZ=[0 for x in range(self.mol.natm)]
#            DZ[atm_idx]=1
#            af=aaff(self.mf,DZ,U=self.U(atm_idx),dP=self.dP(atm_idx),e1=self.e1(atm_idx))
#            af+=alc_deriv_grad_nuc(self.mol,DZ)
#            self.afs[atm_idx]=af
#        return self.afs[atm_idx]
    
    def first_deriv(self,atm_idx):
        return first_deriv_elec(self.mf,self.dV(atm_idx))+first_deriv_nuc_nuc(self.mol,[[atm_idx],[1]])
    
    def second_deriv(self,idx_1,idx_2):
        return second_deriv_elec(self.mf,self.dV(idx_1),self.mo1(idx_2)) +second_deriv_nuc_nuc(self.mol,[[idx_1,idx_2],[1,1]])
    def build_gradient(self,*args):
        if args is None:
            return self.build_gradient(*self.sites)
        idxs=[]
        for arg in args:
            if isinstance(arg,int):
                idxs.append(arg)
        self.gradient=np.asarray([self.first_deriv(x) for x in idxs])
        return self.gradient
    def build_hessian(self,*args):
        if args is None:
            return self.build_hessian(*self.sites)
        mo1s=[]
        dVs=[]
        for arg in args:
            if isinstance(arg,int):
                mo1s.append(self.mo1(arg))
                dVs.append(self.dV(arg))
        mo1s=np.asarray(mo1s)
        dVs=np.asarray(dVs)
        self.hessian=alch_hessian(self.mf,dVs,mo1s) +self.hessian_nuc_nuc(*args)
        return self.hessian
    def hessian_nuc_nuc(self,*args):
            idxs=[]
            for arg in args:
                if isinstance(arg,int):
                    idxs.append(arg)
            hessian=np.zeros((len(idxs),len(idxs)))
            for i in range(len(idxs)):
                for j in range(i,len(idxs)):
                    hessian[i,j]=second_deriv_nuc_nuc(self.mol,[[i,j],[1,1]])/2
            hessian+=hessian.T
            return hessian   
    def build_cubic_hessian(self,*args):
            if args is None:
                return self.build_cubic_hessian(*self.sites)
            idxs=[]
            for arg in args:
                if isinstance(arg,int):
                    idxs.append(arg)
            mo1s=np.asarray([self.mo1(x) for x in idxs])
            dVs=np.asarray([self.dV(x) for x in idxs])
            e1s=np.asarray([self.e1(x) for x in idxs])
            self.cubic_hessian=cubic_alch_hessian(self.mf,dVs,mo1s,e1s)
            return self.cubic_hessian
    def APDFT1(self,pvec):
        return self.mf.e_tot+pvec.dot(self.gradient)
    def APDFT2(self,pvec):
        return self.APDFT1(pvec)+0.5*np.einsum('i,ij,j',pvec,self.hessian,pvec)
    def APDFT3(self,pvec):
        return self.APDFT2(pvec)-1/6*np.einsum('ijk,i,j,k',self.cubic_hessian,pvec,pvec,pvec)     
    
    def targ_energy(self,pvec):
        tmol=FcM_like(self.mol,pvec.tolist())
        b2mf=scf.RHF(tmol)
        return b2mf.scf()
    def targ_mol(self,pvec):
        return FcM_like(self.mol,pvec.tolist())
    
def parse_to_array(natm,dL):
    arr=np.zeros(natm)
    for i in range(len(dL[0])):
        arr[dL[0][i]]=dL[1][i]
    return arr



#class benz_Symm:
#    def __init__(self,mol):
#        self.mol=mol
#        #for our particular benzene molecule 
#        self.axis=np.array((0,0,1))
#        self.irrepr=[0,1]
#        self.eqs={}
#        for eq in range(2,12):
#            self.eqs[eq]={'ref':eq%2,'op':R.from_rotvec(-self.axis*np.pi/3*(eq//2))}
#       
      