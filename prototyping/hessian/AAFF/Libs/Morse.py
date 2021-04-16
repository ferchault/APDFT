import numpy as np    
from berny import Berny, geomlib,Geometry,Math
import berny.coords
import pyscf.data
ang2bohr=1.8897261246
bohr2ang=.5291772109
centimeter2bohr=1.8897261246e+8
plankAU=2*np.pi
lightspeedAU=137.036
dalton_to_au=  1.660e-27 / 9.109e-31

def mpa(g,h,R0,De): # Morse Potential Approximation
    Z=g**2/(2*De)/h
    for i in range(3):
        t=float(np.roots([1,-2,1-2*Z,+Z])[i])
        a=g/(2*De)/(t*(1-t))
        dX=np.log(t)/a
        if a>0 and t>0:
            return R0+dX,a
    return -1

def Morse_V(r,D,a,re):
    return D*(np.exp(-2*a*(r-re))-2*np.exp(-a*(r-re)))+D

def Morse_E(r,r_e,a,De,R0,e0):
    return De*((np.exp(a*(r_e-r))-1)**2-(np.exp(a*(r_e-R0))-1)**2)+e0

def build_h_ic(s,g_ic,h0,B_inv):
    geom0=s.geom.copy()
    B=s.coords.B_matrix(geom0)
    bms=[]
    for i in range(geom0.coords.flatten().shape[0]):
        a=geom0.coords.flatten()
        geom1=geom0.copy()
        a[i]+=.001*bohr2ang
        a=a.reshape(geom0.coords.shape)
        geom1.coords=a
        bms.append((s.coords.B_matrix(geom1)-B)*1000)
    bms_arr=np.asarray(bms)
    BpG2=np.einsum('ijk,j->ik',bms,g_ic)
    h_ic=B_inv.T@(h0-BpG2)@B_inv
    return h_ic
tbbde={"CH":105/627.5,"HC":105/627.5,"NH":110/627.5,"HN":110/627.5,\
       "OH":119/627.5,"HO":119/627.5,"FH":136/627.5,"HF":136/627.5}

def mpa_pb(coords,atoml,g,h,gic=False,solve_ic=False,ghost=[]):
    g=np.asarray(g)
    if not len(h.shape)==2:
        h=h.swapaxes(1,2)
        h=h.reshape(g.shape[0]*3,g.shape[0]*3)
    geom0=Geometry(atoml,coords*bohr2ang)
    bernyobj=Berny(geom0)
    s=bernyobj._state
    B = s.coords.B_matrix(geom0)
    q0=s.coords.eval_geom(geom0)
    B_inv = B.T.dot(Math.pinv(np.dot(B, B.T)))
    if not gic:
        g_ic=np.dot(B_inv.T, (g).reshape(-1))
    else:
        g_ic=g
    h_ic=build_h_ic(s,g_ic,h,B_inv)
    if not solve_ic:    
        return mpa(g_ic[0],h_ic[0,0],q0[0],tbbde[atoml[s.coords._coords[0].i]+atoml[s.coords._coords[0].j]])[0]\
                +g_ic[0]/h_ic[0,0]-q0[0]
    bnr=0
    ddq_mb=np.zeros_like(q0)
    for i in range(len(s.coords._coords)):
        if s.coords._coords[i].__class__ is berny.coords.Bond:
            bnr+=1
            if s.coords._coords[i].i not in ghost and s.coords._coords[i].j not in ghost:
                bondatoms=atoml[s.coords._coords[i].i]+atoml[s.coords._coords[i].j]
                ddq_mb[i]+=mpa(g_ic[i],h_ic[i,i],q0[i],tbbde[bondatoms])[0]+g_ic[i]/h_ic[i,i]-q0[i]
    dq_NR=-np.linalg.solve(h_ic,g_ic)
    ddq_mb[bnr:]=np.linalg.solve(h_ic[bnr:,bnr:],-h_ic[bnr:,:]@(ddq_mb))
    return q0,dq_NR,q0+dq_NR,q0+dq_NR+ddq_mb

def harm_freq(k,an1,an2):
    return to_cm(k,mu(an1,an2))
    
def to_cm(k,Mu):
    return (k/Mu)**0.5/plankAU/lightspeedAU*centimeter2bohr
def mu(an1,an2):
    return (pyscf.data.elements.MASSES[an1]*pyscf.data.elements.MASSES[an2])/ (pyscf.data.elements.MASSES[an1]+pyscf.data.elements.MASSES[an2])*dalton_to_au
def harm_freq(k,an1,an2):
    return to_cm(k,mu(an1,an2))

class Morse_interpolator:
    def __init__(self,g,h,R0,De,e0):
        self.R0=R0
        self.De=De
        self.re,self.a=mpa(g,h,R0,De)
        self.e0=e0
        self.e_min=self.E(self.re)
    def E(self,r):
        return Morse_E(r,self.re,self.a,self.De,self.R0,self.e0)
    def minimum(self):
        return self.re,self.e_min
    def harm_freq_diatomics(self,an1,an2):
        """ input atomic numbers 1 e 2
        """
        ke=2*self.De*self.a**2
        Mu=mu(an1,an2)
        return to_cm(ke,Mu)
    hfd=harm_freq_diatomics
    
    