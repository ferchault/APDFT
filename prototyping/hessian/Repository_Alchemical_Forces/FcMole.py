from pyscf import gto
from AP_utils import parse_charge

NUC_FRAC_CHARGE=gto.mole.NUC_FRAC_CHARGE
NUC_MOD_OF=gto.mole.NUC_MOD_OF
PTR_FRAC_CHARGE=gto.mole.PTR_FRAC_CHARGE



def with_rinv_at_nucleus(self, atm_id):
        rinv = self.atom_coord(atm_id)
        self._env[gto.mole.AS_RINV_ORIG_ATOM] = atm_id  # required by ecp gradients
        return self.with_rinv_origin(rinv)
    
class FracMole(gto.Mole):
    def with_rinv_at_nucleus(self, atm_id):
        rinv = self.atom_coord(atm_id)
        self._env[gto.mole.AS_RINV_ORIG_ATOM] = atm_id  # required by ecp gradients
        return self.with_rinv_origin(rinv)
    
def FcM(fcs=[],**kwargs):
    mol = FracMole()
    mol.build(**kwargs)
    init_charges=mol.atom_charges()
    if fcs:
        fcs=parse_charge(fcs)
        init_charges=mol.atom_charges()
        for j in range(len(fcs[0])):
            mol._env[mol._atm[fcs[0][j],PTR_FRAC_CHARGE]]=init_charges[fcs[0][j]]+fcs[1][j]
            mol._atm[fcs[0][j],NUC_MOD_OF] = NUC_FRAC_CHARGE
        mol.charge=sum(fcs[1])+mol.charge
    return mol

def FcM_like(in_mol,fcs=[]):
    mol=in_mol.copy()
    mol.with_rinv_at_nucleus=with_rinv_at_nucleus.__get__(mol)
    mol.symmetry=None    #symmetry usually breaks after perturbation
    init_charges=mol.atom_charges()
    if fcs:
        fcs=parse_charge(fcs)
        init_charges=mol.atom_charges()
        for j in range(len(fcs[0])):
            mol._env[mol._atm[fcs[0][j],PTR_FRAC_CHARGE]]=init_charges[fcs[0][j]]+fcs[1][j]
            mol._atm[fcs[0][j],NUC_MOD_OF] = NUC_FRAC_CHARGE
        mol.charge=in_mol.charge+sum(fcs[1])
    return mol