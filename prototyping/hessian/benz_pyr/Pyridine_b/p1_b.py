from pyscf import gto, scf
import pyscf.cc
import numpy as np
import pyscf.qmmm
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

calc=scf.RHF(mol)

deltaZ=np.array((1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.))
mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords(), deltaZ)
class NoSelfQMMM(mf.__class__):
    def energy_nuc(self):
        q = mol.atom_charges().astype(np.float)
        q += deltaZ
        return mol.energy_nuc(q)
mf = NoSelfQMMM()
print (mol.energy_nuc())
print (mf.energy_nuc())
hfe=mf.kernel(verbose=0)

cc = pyscf.cc.CCSD(mf).set(frozen = 6,max_cycle=100).run()
dm1 = cc.make_rdm1()
dm1_ao = np.einsum('pi,ij,qj->pq', mf.mo_coeff, dm1, mf.mo_coeff.conj())

np.save('dm',dm1_ao)
with open('CCSD_energy','w') as f:
	f.write (str(cc.e_tot))
