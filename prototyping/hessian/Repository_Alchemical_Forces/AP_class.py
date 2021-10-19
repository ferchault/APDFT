#write down a smart class 

import pyscf.qmmm
from pyscf import gto, scf
import numpy as np
from pyscf import lib
from functools import reduce
from pyscf.scf import cphf
from pyscf import lib
from pyscf.prop.nmr import rhf as rhf_nmr
from aaff import alc_deriv_grad_nuc,aaff_resolv
from scipy.spatial.transform import Rotation as R
from FcMole import FcM_like
from AP_utils import alias_param,parse_charge,DeltaV,charge2symbol 
from alch_deriv import *
from ABSEC import abse_atom
import copy


class APDFT_perturbator(lib.StreamObject):
    @alias_param(param_name="symmetry", param_alias='symm')
    def __init__(self,mf,symmetry=None,sites=None):
        self.mf=mf
        self.mol=mf.mol
        self.symm=symmetry
        self.sites=[]
        for site in sites: self.sites.append(site)
        self.DeltaV=DeltaV
        self.alchemy_cphf_deriv=alchemy_cphf_deriv
        self.make_dP=make_dP
        self.make_U=make_U
        self.dVs={}
        self.mo1s={}
        self.e1s={}
        self.dPs={}
        self.afs={}
        self.perturb()
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
            if site in self.mo1s: 
                pass
            elif  self.symm and site in self.symm.eqs:
                ref_idx=self.symm.eqs[site]['ref']
                if ref_idx in self.mo1s:
                    self.dVs[site]=DeltaV(self.mol,[[site],[1]])
                    self.mo1s[site],self.e1s[site]=self.symm.rotate_mo1e1(self.mo1s[ref_idx],self.e1s[ref_idx],\
                    site,ref_idx,self.mf.mo_coeff,self.mf.get_ovlp())
                else: continue
            else:
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
        
    def af(self,atm_idx):
        if atm_idx in self.afs: 
            return self.afs[atm_idx] 
        elif self.symm and atm_idx in self.symm.eqs:
            ref_idx=self.symm.eqs[atm_idx]['ref']
            afr=self.af(ref_idx)
            self.afs[atm_idx]=self.symm.symm_gradient(afr,atm_idx,ref_idx)
        else:
            if atm_idx not in self.sites:
                self.sites.append(atm_idx)
                self.perturb()
            DZ=[0 for x in range(self.mol.natm)]
            DZ[atm_idx]=1
            af=aaff_resolv(self.mf,DZ,U=self.U(atm_idx),dP=self.dP(atm_idx),e1=self.e1(atm_idx))
            af+=alc_deriv_grad_nuc(self.mol,DZ)
            self.afs[atm_idx]=af
        return self.afs[atm_idx]
    
    def first_deriv(self,atm_idx):
        return first_deriv_elec(self.mf,self.dV(atm_idx))+first_deriv_nuc_nuc(self.mol,[[atm_idx],[1]])
    
    def second_deriv(self,idx_1,idx_2):
        return second_deriv_elec(self.mf,self.dV(idx_1),self.mo1(idx_2)) +second_deriv_nuc_nuc(self.mol,[[idx_1,idx_2],[1,1]])
    def third_deriv(self,pvec):
        pvec=np.asarray(pvec)
        return np.einsum('ijk,i,j,k',self.cubic_hessian,pvec,pvec,pvec)
    def build_gradient(self):
        idxs=self.sites
        self.gradient=np.asarray([self.first_deriv(x) for x in idxs])
        return self.gradient
    def build_hessian(self):
        mo1s=[]
        dVs=[]
        for id in self.sites:
            mo1s.append(self.mo1(id))
            dVs.append(self.dV(id))
        mo1s=np.asarray(mo1s)
        dVs=np.asarray(dVs)
        self.hessian=alch_hessian(self.mf,dVs,mo1s) +self.hessian_nuc_nuc(*self.sites)
        return self.hessian
    def hessian_nuc_nuc(self,*args):
            idxs=[]
            for arg in args:
                if isinstance(arg,int):
                    idxs.append(arg)
            hessian=np.zeros((len(idxs),len(idxs)))
            for i in range(len(idxs)):
                for j in range(i,len(idxs)):
                    hessian[i,j]=second_deriv_nuc_nuc(self.mol,[[idxs[i],idxs[j]],[1,1]])/2
            hessian+=hessian.T
            return hessian   
    def build_cubic_hessian(self):
            idxs=self.sites
            mo1s=np.asarray([self.mo1(x) for x in idxs])
            dVs=np.asarray([self.dV(x) for x in idxs])
            e1s=np.asarray([self.e1(x) for x in idxs])
            self.cubic_hessian=cubic_alch_hessian(self.mf,dVs,mo1s,e1s)
            return self.cubic_hessian
    def build_all(self):
        self.build_gradient(*self.sites)
        self.build_hessian(*self.sites)
        self.build_cubic_hessian(*self.sites)
    def APDFT1(self,pvec):
        pvec=np.asarray(pvec)
        return self.mf.e_tot+pvec.dot(self.gradient)
    def APDFT2(self,pvec):
        pvec=np.asarray(pvec)
        return self.APDFT1(pvec)+0.5*np.einsum('i,ij,j',pvec,self.hessian,pvec)
    def APDFT3(self,pvec):
        pvec=np.asarray(pvec)
        return self.APDFT2(pvec)+1/6*np.einsum('ijk,i,j,k',self.cubic_hessian,pvec,pvec,pvec)     
    
    def target_energy_ref_bs(self,pvec):  # with refernce basis set 
        tmol=self.target_mol_ref_bs(pvec)
        b2mf=scf.RHF(tmol)
        return b2mf.scf(dm0=b2mf.init_guess_by_1e())
    def target_mol_ref_bs(self,pvec):
        if type(pvec) is list:
            tmol=FcM_like(self.mol,fcs=pvec)
        else:
            tmol=FcM_like(self.mol,fcs=pvec.tolist())
        return tmol

    def target_mol(self,pvec):
        splitted=(self.mol.atom.split())
        refchgs=copy.deepcopy(self.mol.atom_charges())
        for idx in range(len(pvec)):
            refchgs[self.sites[idx]]+=int(pvec[idx])
        for idx in range(len(pvec)):
            splitted[self.sites[idx]*4]=charge2symbol[refchgs[self.sites[idx]]]
        atomstr=" ".join(splitted)
        tmol=gto.M(atom=atomstr,unit=self.mol.unit,basis=self.mol.basis,charge=self.mol.charge+sum(pvec))
        return tmol

    def target_energy(self,pvec):
        tmf=scf.RHF(self.target_mol(pvec))
        return tmf.scf()

    
    def ap_bsec(self,pvec):
        ral=[charge2symbol[i] for i in self.mol.atom_charges()]
        tal=[charge2symbol[i] for i in self.target_mol(pvec).atom_charges()]
        if len(ral) != len(tal):
            print(ral,tal,"reference and target lengths do not match!", sys.exc_info()[0])
            raise 
        bsecorr=0
        for i in range(len(ral)):
            bsecorr+=abse_atom(ral[i],tal[i],bs=self.mol.basis)
        return bsecorr

def parse_to_array(natm,dL):
    arr=np.zeros(natm)
    for i in range(len(dL[0])):
        arr[dL[0][i]]=dL[1][i]
    return arr

        