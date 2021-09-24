from pyscf import gto,scf

class ao_RHF(scf.rhf.RHF):
    def __init__(self,mol,S_ao=None,hcore=None,eri=None):
        super().__init__(mol)
        self.S_ao=S_ao
        self.hcore=hcore
        self._eri=eri
    
    def get_hcore(self,mol=None):
        if self.hcore is not None:
            return self.hcore 
        else:
            return super().get_hcore(mol)

    def get_ovlp(self,mol=None):
        if self.S_ao is not None:
            return self.S_ao 
        else:
            return super().get_ovlp(mol)
