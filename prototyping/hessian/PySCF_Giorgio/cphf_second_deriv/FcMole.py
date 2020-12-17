from pyscf import gto

NUC_FRAC_CHARGE=gto.mole.NUC_FRAC_CHARGE
NUC_MOD_OF=gto.mole.NUC_MOD_OF
PTR_FRAC_CHARGE=gto.mole.PTR_FRAC_CHARGE

class FracMole(gto.Mole):
    def with_rinv_at_nucleus(self, atm_id):
        rinv = self.atom_coord(atm_id)
        self._env[gto.mole.AS_RINV_ORIG_ATOM] = atm_id  # required by ecp gradients
        return self.with_rinv_origin(rinv)
    
def FcM(fcs=[],**kwargs):
    """ There are two options: 
    1) call FcM(**kwargs,fcs=[c1,c2,--cn]) with a list of length equal to the number of atoms
    2) FcM(**kwargs,fcs=[[aidx1,aidx2,..,aidxn],[c1,c2,..cn]]) with a list of two sublist for atoms' indexes and fract charges
    """
    mol = FracMole()
    mol.build(**kwargs)
    init_charges=mol.atom_charges().copy()
    if len(fcs)==mol.natm and fcs[0].__class__ in (float,int):
        print("prima opzione")
        mol._atm[:,NUC_MOD_OF] = NUC_FRAC_CHARGE
        for i in range (mol.natm):
            mol._env[mol._atm[i,PTR_FRAC_CHARGE]]=init_charges[i]+fcs[i]
        return mol
    if len(fcs)==2 and len(fcs[0])==len(fcs[1]):
        for j in range(len(fcs[0])):
            mol._env[mol._atm[fcs[0][j],PTR_FRAC_CHARGE]]=init_charges[fcs[0][j]]+fcs[1][j]
            mol._atm[fcs[0][j],NUC_MOD_OF] = NUC_FRAC_CHARGE
        return mol
    else:
        print("can not parse fractional charges")
        return 1
    return mol
