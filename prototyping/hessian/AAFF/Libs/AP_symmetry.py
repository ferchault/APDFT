import numpy as np
from scipy.spatial.transform import Rotation as R


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
