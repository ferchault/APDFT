#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:51:43 2019

@author: giorgiod


#########  ATOMO DI CARBONIO COME ESEMPIO PER MATRICI & OPERATORI  ########

"""

from horton import*
import numpy as np
from numpy import dot

log.set_level(1)
natoms=np.array([6])
coordinates=np.array([[-1.,0.,0.]])

mol=IOData(title='CO')
mol.numbers=natoms
mol.coordinates=coordinates
mol.pseudo_numbers=np.asarray([6.])

obasis=get_gobasis(mol.coordinates,mol.numbers,'6-31g')

lf=DenseLinalgFactory(obasis.nbasis)

#orbital integrals
olp=obasis.compute_overlap(lf)
S=olp._array
kin=obasis.compute_kinetic(lf)
T=kin._array
na=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
V=na._array
er=obasis.compute_electron_repulsion(lf)
R=er._array

exp_alpha=lf.create_expansion()
exp_beta=lf.create_expansion()
print exp_alpha.get_homo_energy()
guess_core_hamiltonian(olp, kin, na, exp_alpha,exp_beta)
print exp_alpha.get_homo_energy()

occ_model=AufbauOccModel(3,3)
occ_model.assign(exp_alpha, exp_beta)
 
external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
terms = [
    UTwoIndexTerm(kin, 'kin'),
    UDirectTerm(er, 'hartree'),
    UExchangeTerm(er, 'x_hf'),
    UTwoIndexTerm(na, 'ne'),
]
ham = UEffHam(terms, external)
dm_alpha=exp_alpha.to_dm()
dm_beta=exp_beta.to_dm()

scf_solver = EDIIS2SCFSolver(1e-6,maxiter=300)
scf_solver(ham, lf, olp, occ_model, dm_alpha, dm_beta)

for x in ham.cache.iterkeys():    
    print x, "         ", ham.cache[x]

Db=dm_beta
db=Db._array
Da=dm_alpha
da=Da._array
Df= ham.cache['dm_full']
df=Df._array
F=er.contract_two_to_two('abcd,ac->bd',Df);F.iadd(na);F.iadd(kin)
Fa=F.copy();Fb=F.copy()
Fa.iadd(er.contract_two_to_two('abcd,ad->bc',Da),-1.)
Fb.iadd(er.contract_two_to_two('abcd,ad->bc',Db),-1.)

exp_alpha.from_fock(Fa.copy(),olp.copy())

#######  Come faccio ad ottenere le energie degli orbitali ?????????????/


###  exp_alpha.to_dm() ===  np.dot(exp_alpha._coeffs*exp_alpha.occupations, exp_alpha._coeffs.T)

""" the keywords in the cache of the hamiltonian:
    for x in ham.cache.iterkeys():
    print x
    
energy_x_hf           -5.02955076609
energy_hartree           17.8173025071
dm_full           <horton.matrix.dense.DenseTwoIndex object at 0x7f1a04033e90>
dm_beta           <horton.matrix.dense.DenseTwoIndex object at 0x7f1a04033dd0>
energy_ne           -87.973169153
energy           -37.5882040184
op_x_hf_alpha           <horton.matrix.dense.DenseTwoIndex object at 0x7f1a04033f90>
op_hartree           <horton.matrix.dense.DenseTwoIndex object at 0x7f1a04033f10>
dm_alpha           <horton.matrix.dense.DenseTwoIndex object at 0x7f1a04033b10>
energy_kin           37.5972133937
op_x_hf_beta           <horton.matrix.dense.DenseTwoIndex object at 0x7f1a04033ad0>
energy_nn           0.0


da.contract_two('ab,ba',na)

da.contract_two('ab,ba',na)+db.contract_two('ab,ba',na)

da.contract_two('ab,ab',na)+db.contract_two('ab,ab',na)

er.__class__
Out[89]: horton.matrix.dense.DenseFourIndex


da.contract_two('ab,ab',kin)+db.contract_two('ab,ab',kin)
Out[90]: 110.24122218613226
 c1=er.contract_two_to_two('abcd,ac->bd',dmf)

In [92]: c1.contract_two('ab,ba',dmf)
Out[92]: 37.5603759557081

In [93]: c1.contract_two('ab,ba',dmf)/2
Out[93]: 18.78018797785405




 Df.contract_two('ab,ab',er.contract_two_to_two('abcd,ac->bd',Df))
Out[14]: 35.63460501419172

 Df.contract_two('ab,ab',er.contract_two_to_two('abcd,ac->bd',Df))/2
Out[15]: 17.81730250709586

 Da.contract_two('ab,ab',er.contract_two_to_two('abcd,ac->bd',Db))/2
Out[16]: 4.454325626773965

 Da.contract_two('ab,ab',er.contract_two_to_two('abcd,ac->bd',Da))/2
Out[17]: 4.454325626773965

 Db.contract_two('ab,ab',er.contract_two_to_two('abcd,ac->bd',Db))/2
Out[18]: 4.454325626773965
"""




