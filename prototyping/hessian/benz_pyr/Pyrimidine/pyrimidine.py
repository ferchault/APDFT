from pyscf import gto, scf
import pyscf.cc
import numpy as np
import time 

time_0=time.time()
pyrimidine_atom="""N 0.0 2.6196127020137916 0.0;
C 2.268651148020338 1.309806351006896 0.0;
H 4.028886797087305 2.326078876832885 0.0;
N 2.2686511480203384 -1.3098063510068951 0.0;
C 3.208100310526939e-16 -2.6196127020137916 0.0;
H 5.697250102155325e-16 -4.652157753665769 0.0;
C -2.2686511480203375 -1.309806351006897 0.0;
H -4.028886797087303 -2.3260788768328866 0.0;
C -2.2686511480203393 1.309806351006894 0.0;
H -4.028886797087306 2.3260788768328813 0.0
"""
mol = gto.M(atom=pyrimidine_atom, basis='def2-TZVP',unit='bohr')

calc=scf.RHF(mol)
hfe=calc.kernel()
cc = pyscf.cc.CCSD(calc).set(frozen = 6,max_cycle=100).run()
dm1 = cc.make_rdm1()
dm1_ao = np.einsum('pi,ij,qj->pq', calc.mo_coeff, dm1, calc.mo_coeff.conj())

time_F=time.time()
np.save('dm',dm1_ao)
with open('CCSD_energy','w') as f:
	f.write (str(cc.e_tot))
	f.write ('   total time(s)='+str(time_F-time_0 ) )
