from pyscf import gto, scf
import pyscf.cc
import numpy as np
angstrom = 1 / 0.52917721067

benz_atom="""C 0.0 2.6196127020137916 0.0;
H 0.0 4.652157753665769 0.0;
C 2.268651148020338 1.309806351006896 0.0;
H 4.028886797087305 2.326078876832885 0.0;
C 2.2686511480203384 -1.3098063510068951 0.0;
H 4.028886797087305 -2.3260788768328835 0.0;
C 3.208100310526939e-16 -2.6196127020137916 0.0;
H 5.697250102155325e-16 -4.652157753665769 0.0;
C -2.2686511480203375 -1.309806351006897 0.0;
H -4.028886797087303 -2.3260788768328866 0.0;
C -2.2686511480203393 1.309806351006894 0.0;
H -4.028886797087306 2.3260788768328813 0.0
"""
mol = gto.M(atom=benz_atom, basis='def2-TZVP',unit='bohr')

def DeltaV(mol,dL,on_atoms):
    mol.set_rinv_orig_(mol.atom_coords()[on_atoms[0]]/angstrom)
    dV=mol.intor('int1e_rinv')*dL[0]
    for i in range(1,len(dL)):
    	mol.set_rinv_orig_(mol.atom_coords()[on_atoms[i]]/angstrom)
    	dV+=mol.intor('int1e_rinv')*dL[i]
    return dV

dV1=DeltaV(mol,[1,-1],[0,1])

dV2=DeltaV(mol,[1,-1,1.,-1,],[0,1,4,5])

dV3=DeltaV(mol,[1,-1,1.,-1.,1.,-1.],[0,1,4,5,8,9])

np.save('dV1',dV1)

np.save('dV2',dV2)
np.save('dV3',dV3)
 
