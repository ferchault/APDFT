import numpy as np
from scipy.spatial.transform import Rotation as R
from pyscf.symm.basis import _ao_rotation_matrices as aorm

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


class benz_Symm:
    def __init__(self,mol):
        self.mol=mol
        #for our particular benzene molecule 
        self.axis=np.array((0,0,1))
        self.irrepr=[0,1]
        self.eqs={}
        for eq in range(2,12):
            self.eqs[eq]={'ref':eq%2,'op':R.from_rotvec(-self.axis*np.pi/3*(eq//2))}
    
    def symm_gradient(self,afr,atm_idx,ref_idx):
        return rotate_grad(self.eqs[atm_idx]['op'].apply(afr),atm_idx,ref_site=ref_idx)

    def rotate_mo1e1(self,mo1,e1,site,ref_idx,C,S):
        nocc=self.mol.nelec[0]
        rm=self.make_RM(site,ref_idx)
        mo1r=(C.T@S@rotate_matrix(rm.T@(C@mo1@C.T[:nocc,:])@rm,self.mol,site,ref_site=ref_idx)@S@C)[:,:nocc]
        e1r=(C.T@S@rotate_matrix(rm.T@(C[:,:nocc]@e1@C.T[:nocc,:])@rm,self.mol,site,ref_site=ref_idx)@S@C)[:nocc,:nocc]
        return (mo1r,e1r)

    def make_RM(self,site,ref_idx):
        p_idxs=[i for i,elem in enumerate(self.mol.ao_labels()) if  "px" in elem]
        d_idxs=[i for i,elem in enumerate(self.mol.ao_labels()) if  "dxy" in elem]
        f_idxs=[i for i,elem in enumerate(self.mol.ao_labels()) if  "fy^3" in elem]
        rm_p= self.eqs[site]['op'].as_dcm()
        Dm_ao=aorm(self.mol,rm_p)
        rm=np.eye(self.mol.nao)
        for i in p_idxs:
            rm[i:i+3,i:i+3]=Dm_ao[1]
        for i in d_idxs:
            rm[i:i+5,i:i+5]=Dm_ao[2]
        for i in f_idxs:
            rm[i:i+7,i:i+7]=Dm_ao[3]
        return rm





# for methane stuff
def rothess(h):
    hr=np.zeros_like(h)
    ridx={0:0,1:1,2:3,3:4,4:2}
    for i in range(5):
        for j in range(5):
            hr[i,j]=r.apply(r.apply(h[ridx[i],ridx[j]]).T).T
    return hr
def rotgrad(g):
    b=r.apply(g)
    b[[2,3,4]]=b[[3,4,2]]
    return b 
