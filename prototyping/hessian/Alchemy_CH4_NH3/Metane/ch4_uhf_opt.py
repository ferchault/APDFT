# -*- coding: utf-8 -*-
from horton import *
import numpy as np
from numpy.linalg import norm
from numpy import dot,cross
mol=IOData.from_file('co.xyz')
log.set_level(1)
#print Methane.numbers
#print mol.coordinates
#mol.coordinates*=angstrom
#print mol.coordinates
def uhf (mol,basis_set='6-31g',get_dm=False):
    obasis=get_gobasis(mol.coordinates,mol.numbers,basis_set)
    lf=DenseLinalgFactory(obasis.nbasis)
    #orbital integrals
    olp=obasis.compute_overlap(lf)
    kin=obasis.compute_kinetic(lf)
    na=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
    er=obasis.compute_electron_repulsion(lf)
    
    exp_alpha=lf.create_expansion()
    exp_beta=lf.create_expansion()

    guess_core_hamiltonian(olp, kin, na, exp_alpha,exp_beta)
    occ_model=AufbauOccModel(5,5)
    occ_model.assign(exp_alpha, exp_beta)
    
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        UTwoIndexTerm(kin, 'kin'),
        UDirectTerm(er, 'hartree'),
        UExchangeTerm(er, 'x_hf'),
        UTwoIndexTerm(na, 'ne'),
    ]
    ham = UEffHam(terms, external)
    
    dm_alpha = exp_alpha.to_dm()
    dm_beta = exp_beta.to_dm()
    # - SCF solver
    scf_solver = EDIIS2SCFSolver(1e-10,maxiter=200)
    scf_solver(ham, lf, olp, occ_model, dm_alpha, dm_beta)
    if get_dm:
        return dm_alpha,dm_beta,ham.cache['dm_full']
    else:
        return ham.cache['energy']

def der_HF(mol):
    grid=BeckeMolGrid(mol.coordinates,mol.numbers,random_rotate=False,agspec='insane')
    df=uhf(mol,get_dm=True)[2]
    obasis=get_gobasis(mol.coordinates,mol.numbers,'6-31g')
    rho= obasis.compute_grid_density_dm(df, grid.points)
    gradient=[]
    print "Natoms=", grid.integrate(rho)
    #calculate the analytical force on the atoms (erroneous Hellman-Feynman approximation )
    for x in range(mol.natom):
        force=np.zeros(3)
        dRx=grid.points[:,0]-mol.coordinates[x][0]
        dRy=grid.points[:,1]-mol.coordinates[x][1]
        dRz=grid.points[:,2]-mol.coordinates[x][2] 
        mod_R=(dRx[:]**2+dRy[:]**2+dRz[:]**2)**1.5   ## =  |R-r|^3
    
        force[0]=grid.integrate(rho,dRx/mod_R)*mol.pseudo_numbers[x]
        force[1]=grid.integrate(rho,dRy/mod_R)*mol.pseudo_numbers[x]
        force[2]=grid.integrate(rho,dRz/mod_R)*mol.pseudo_numbers[x]
        for y in force:
            gradient.append(y)
    g=np.asarray(gradient)
    return g

def der_NN(mol):
    h=1e-9
    nn0=compute_nucnuc(mol.coordinates, mol.pseudo_numbers)
    g=[]
    for x in range(mol.natom):
        for y in range(3):
            mol.coordinates[x][y]+=h
            g.append(compute_nucnuc(mol.coordinates, mol.pseudo_numbers))
            mol.coordinates[x][y]-=h
    g=np.asarray(g)
    g-=nn0
    g/=h
    return g
    
def der_WFF(mol,basis_set='6-31g'):
    da,db,df=uhf(mol,basis_set='6-31g',get_dm=True)
    obasis=get_gobasis(mol.coordinates,mol.numbers,basis_set)
    lf=DenseLinalgFactory(obasis.nbasis)
    g=[]
    gdb=[]
    olp0=obasis.compute_overlap(lf)
    kin0=obasis.compute_kinetic(lf)
    na0=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
    er0=obasis.compute_electron_repulsion(lf)
    
    F=er0.contract_two_to_two('abcd,ac->bd',df);F.iadd(na0);F.iadd(kin0)
    Fa=F.copy();Fb=F.copy()
    Fa.iadd(er0.contract_two_to_two('abcd,ad->bc',da),-1.)
    Fb.iadd(er0.contract_two_to_two('abcd,ad->bc',db),-1.)
    
    exp_alpha=lf.create_expansion()
    exp_beta=lf.create_expansion()
    exp_alpha.from_fock_and_dm(Fa.copy(),da,olp0.copy())
    exp_beta.from_fock_and_dm(Fb.copy(),db,olp0.copy())
    
#    ### one electron operators ++++++++++
#    E0=df.contract_two('ab,ab',na0)+df.contract_two('ab,ab',kin0)
#    #### two electron operatprs +++++++++  Coulontract_two('ab,ab',df)*0.5
#    E0-=er0.contract_two_to_two('abcd,ab->cd',da).contract_two('ab,ab',da)*0.5
#    E0-=er0.contract_two_to_two('abcd,ab->cd',db).contract_two('ab,ab',db)*0.5

    h=1e-5
    for x in range(mol.coordinates.shape[0]):
        for y in range(3):
            mol.coordinates[x][y]+=h
            obasis=get_gobasis(mol.coordinates,mol.numbers,basis_set)
            
            lf=DenseLinalgFactory(obasis.nbasis)
            olp1=obasis.compute_overlap(lf)
            olp1.iadd(olp0,-1.)
            kin1=obasis.compute_kinetic(lf)
            kin1.iadd(kin0,-1.)
            na1=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
            na1.iadd(na0,-1.)
            er1=obasis.compute_electron_repulsion(lf)
            er1.iadd(er0,-1.)

            dE=df.contract_two('ab,ab',na1)+df.contract_two('ab,ab',kin1)
            #### two electron operatprs +++++++++  Coulomb and Hartree Fock exchange 
            dE+=er1.contract_two_to_two('abcd,ac->bd',df).contract_two('ab,ab',df)*0.5
            dE-=er1.contract_two_to_two('abcd,ab->cd',da).contract_two('ab,ab',da)*0.5
            dE-=er1.contract_two_to_two('abcd,ab->cd',db).contract_two('ab,ab',db)*0.5                        
            

            eWDa=dot(exp_alpha._coeffs*exp_alpha._energies*exp_alpha.occupations,exp_alpha._coeffs.T)
            dE-=np.sum(eWDa*olp1._array)
            
            eWDb=dot(exp_beta._coeffs*exp_beta._energies*exp_beta.occupations,exp_beta._coeffs.T)
            dE-=np.sum(eWDb*olp1._array)            
            ### one electron operators ++++++++++
            

            mol.coordinates[x][y]-=h
            g.append(dE/h)
    gradient=np.asarray(g)
    gradient+=der_NN(mol)
    return gradient 



def print_dist():
    for x in mol.coordinates[1:]:
        print norm(x-mol.coordinates[0])

def bad_opt():
    conv=False
    while not conv:
        eg=uhf(mol)
        g=np.asarray(eg[1])     
        print eg[0],'-------',norm(g), g[1]
        print_dist()
        if norm(g)<1e-3:
            conv=True
        else:
            mol.coordinates+=np.asarray([g[3*i:3*i+3]for i in range(mol.coordinates.size//3)])/norm(g)*0.0001    


        
def der_SCF(mol):
    eng=[]
    h=1e-5
    e0=uhf(mol)
    for x in mol.coordinates:
        x+=[h,0,0]
        eng.append(uhf(mol))
        x+=[-2*h,0,0]
        eng.append(uhf(mol))
        x+=[h,0,0]
        
        x+=[0,h,0]
        eng.append(uhf(mol))
        x+=[0,-2*h,0]
        eng.append(uhf(mol))
        x+=[0,h,0]
        
        x+=[0,0,h]
        eng.append(uhf(mol))
        x+=[0,0,-2*h]
        eng.append(uhf(mol))
        x+=[0,0,h]
    eng=np.asarray(eng)
    g=(eng[0::2]-eng[1::2])/2./h
    return g


def grad_opt():
    conv=False
    while not conv:
        eg=uhf(mol)
        g=np.asarray(eg[1])     
        dx=np.asarray([g[3*i:3*i+3]for i in range(mol.coordinates.size//3)])/norm(g)*1.e-3
        mol.coordinates+=dx
        eg2=uhf(mol)
        g2=np.asarray(eg2[1])
        g2=np.dot(g,g2)/(norm(g))**2*g          #projection on g
        DX=-norm(g2)/(norm(g2)-norm(g))
        mol.coordinates+=dx*DX
        if norm(g)<1e-3:
            conv=True
        print eg[0],'-------',norm(g), g[1],np.sum(g)
        print_dist()   


def grad_opt2(mol):
    conv=False
    while not conv:
        g=gradient_from_scf(mol)
        dx=np.asarray([g[3*i:3*i+3]for i in range(mol.coordinates.size//3)])/norm(g)*1.e-3
        mol.coordinates+=dx
        g2=gradient_from_scf(mol)
        g2=np.dot(g,g2)/(norm(g))**2*g          #projection on g
        DX=-norm(g2)/(norm(g2)-norm(g))
        
        mol.coordinates+=dx*DX
        if norm(g)<1e-3:
            conv=True
        print uhf(mol,compute_grad=False),'-------',norm(g), g[1],np.sum(g)
        print_dist()




def der_WFFold(mol,basis_set='6-31g'):
    da,db,df=uhf(mol,basis_set='6-31g',get_dm=True)
    obasis=get_gobasis(mol.coordinates,mol.numbers,basis_set)
    lf=DenseLinalgFactory(obasis.nbasis)
    g=[]
    gdb=[]
    olp0=obasis.compute_overlap(lf)
    kin0=obasis.compute_kinetic(lf)
    na0=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
    er0=obasis.compute_electron_repulsion(lf)
    
    ### one electron operators ++++++++++
    E0=df.contract_two('ab,ab',na0)+df.contract_two('ab,ab',kin0)
    #### two electron operatprs +++++++++  Coulomb and Hartree Fock exchange 
    E0+=er0.contract_two_to_two('abcd,ac->bd',df).contract_two('ab,ab',df)*0.5
    E0-=er0.contract_two_to_two('abcd,ab->cd',da).contract_two('ab,ab',da)*0.5
    E0-=er0.contract_two_to_two('abcd,ab->cd',db).contract_two('ab,ab',db)*0.5
    
    h=1e-5
    for x in range(mol.coordinates.shape[0]):
        for y in range(3):
            mol.coordinates[x][y]+=h
            obasis=get_gobasis(mol.coordinates,mol.numbers,basis_set)
            
            lf=DenseLinalgFactory(obasis.nbasis)
            olp1=obasis.compute_overlap(lf)
            kin1=obasis.compute_kinetic(lf)
            na1=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
            er1=obasis.compute_electron_repulsion(lf)
            ### one electron operators ++++++++++
            E1=df.contract_two('ab,ab',na1)+df.contract_two('ab,ab',kin1)
            #### two electron operatprs +++++++++  Coulomb and Hartree Fock exchange 
            E1+=er1.contract_two_to_two('abcd,ac->bd',df).contract_two('ab,ab',df)*0.5
            E1-=er1.contract_two_to_two('abcd,ab->cd',da).contract_two('ab,ab',da)*0.5
            E1-=er1.contract_two_to_two('abcd,ab->cd',db).contract_two('ab,ab',db)*0.5 
            print (E1-E0)/h
            
            ## Idempotence condition  dD =-D*dS*D
            olp1.iadd(olp0,-1.)
            ddf=df.contract_two_to_two('ab,bc->ac',olp1.contract_two_to_two('ab,bc->ac',df))
            dda=da.contract_two_to_two('ab,bc->ac',olp1.contract_two_to_two('ab,bc->ac',da))
            ddb=db.contract_two_to_two('ab,bc->ac',olp1.contract_two_to_two('ab,bc->ac',db))
            
#            ddf._array*=-1.
#            dda._array*=-1.
#            ddb._array*=-1.
            
            Ed=ddf.contract_two('ab,ab',na0) 
            Ed+=ddf.contract_two('ab,ab',kin0)
            Ed+=er0.contract_two_to_two('abcd,ac->bd',df).contract_two('ab,ab',ddf)/2
            Ed+=er0.contract_two_to_two('abcd,ac->bd',ddf).contract_two('ab,ab',df)/2
            
            Ed-=er0.contract_two_to_two('abcd,ad->bc',dda).contract_two('ab,ab',da)/2
            Ed-=er0.contract_two_to_two('abcd,ad->bc',da).contract_two('ab,ab',dda)/2
            Ed-=er0.contract_two_to_two('abcd,ad->bc',db).contract_two('ab,ab',ddb)/2
            Ed-=er0.contract_two_to_two('abcd,ad->bc',ddb).contract_two('ab,ab',db)/2
            print 'ED   ',Ed/h
            
            mol.coordinates[x][y]-=h
            g.append((E1+Ed-E0)/h)
    gradient=np.asarray(g)

    return gradient 
