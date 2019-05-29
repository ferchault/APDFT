# -*- coding: utf-8 -*-
from horton import *
import numpy as np
from numpy.linalg import norm
from numpy import dot,cross
mol=IOData.from_file('Methane.xyz')
log.set_level(1)
#print Methane.numbers
#print mol.coordinates
#mol.coordinates*=angstrom
#print mol.coordinates
def uhf (mol,basis_set='6-31g',compute_grad=True):
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
    
    grid=BeckeMolGrid(mol.coordinates,mol.numbers,random_rotate=False,agspec='insane')
    rho_alpha = obasis.compute_grid_density_dm(dm_alpha, grid.points)
    rho_beta = obasis.compute_grid_density_dm(dm_beta, grid.points)
    rho=rho_alpha+rho_beta
    gradient=[]
    
    #calculate the analytical force on the atoms
    if compute_grad==True:
        for x in range(mol.natom):
            force=np.zeros(3)
            dRx=grid.points[:,0]-mol.coordinates[x][0]
            dRy=grid.points[:,1]-mol.coordinates[x][1]
            dRz=grid.points[:,2]-mol.coordinates[x][2]
            mod_R=(dRx[:]**2+dRy[:]**2+dRz[:]**2)**1.5
        
            force[0]=grid.integrate(rho,dRx/mod_R)*mol.pseudo_numbers[x]
            force[1]=grid.integrate(rho,dRy/mod_R)*mol.pseudo_numbers[x]
            force[2]=grid.integrate(rho,dRz/mod_R)*mol.pseudo_numbers[x]
            
            #add nuclear repulsion contribution
            for i in range(mol.natom) :
                if i!=x:
                    force+=mol.pseudo_numbers[i]*mol.pseudo_numbers[x]*\
                    (mol.coordinates[x]-mol.coordinates[i])/norm(mol.coordinates[x]-mol.coordinates[i])**3
            for y in force:
                gradient.append(y)
        g=np.asarray(gradient)
        return ham.cache['energy'],g
    else:
        return ham.cache['energy']
def comp_gradient(mol,rho):
    grid=BeckeMolGrid(mol.coordinates,mol.numbers,random_rotate=False)
    
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
        
def gradient_from_scf(mol):
    eng=[]
    h=.001
    for x in mol.coordinates:
        x+=[h,0,0]
        eng.append(uhf(mol,compute_grad=False))
        x+=[-2*h,0,0]
        eng.append(uhf(mol,compute_grad=False))
        x+=[h,0,0]
        
        x+=[0,h,0]
        eng.append(uhf(mol,compute_grad=False))
        x+=[0,-2*h,0]
        eng.append(uhf(mol,compute_grad=False))
        x+=[0,h,0]
        
        x+=[0,0,h]
        eng.append(uhf(mol,compute_grad=False))
        x+=[0,0,-2*h]
        eng.append(uhf(mol,compute_grad=False))
        x+=[0,0,h]
    eng=np.asarray(eng)
    g=(eng[0::2]-eng[1::2])/2./h
    return g

def grad_opt2():
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


#grad_opt()
#for x in mol.coordinates:
#    x+=[0.05,0,0]
#    e.append(uhf(mol)[0])
#    x+=[-0.05,0,0]
#    x+=[0,0.05,0]
#    e.append(uhf(mol)[0])
#    x+=[0,-0.05,0]
#    x+=[0,0,0.05]
#    e.append(uhf(mol)[0])
#    x+=[0,0,-0.05]  

    
#r.derive(0,.001)
#
#c1=r.Internal
#print c1
#print r.derive(1,.001)
#print c1
#print r.Internal
#print r.derive(2,.01)


