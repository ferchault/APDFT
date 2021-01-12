import ase.units as au
from ase import Atoms
import numpy as np
import os
import shutil
import sys
import subprocess

sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/apdft-ofdft-cpmd/')
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies/')

import profess_io as pio
from explore_qml_data import get_num_val_elec

class DensityOptimizer():
    
    def __init__(self):
        pass
    
    def initialize(self, atoms, dt, mu, profess_path, workdir):
        self.workdir = workdir
        self.profess_path = profess_path # path to PROFESS executable
        
        self.fs2atu = 1/(au._aut*1e15) # converts time from fs to a.t.u.: 1 fs * fs2atu = time in a.t.u.
        self.dt = dt/au.fs * self.fs2atu # conversion back to fs from ase time format by dividing by au.fs then conversion from fs to a.t.u.
        self.mu = mu
        self.energies = []
        
        self.atoms = atoms
        self.V = self.atoms.get_volume()/au.Bohr**3
        self.Ne = - atoms.get_initial_charges().sum() + get_num_val_elec(atoms.get_atomic_numbers())
        

        
    def calculate_dEdX(self, density_file):
        """
        calculates dEdX for the density specified in density file
        """
        # copy density file to work_dir/density
        shutil.copy(density_file, os.path.join(self.workdir, 'density'))
        # run PROFESS-1
        self.p = self.run_profess()
        assert self.p.returncode == 0, 'Calculation of forces failed'
        
    def calculate_lambda(self):
        #scaling_factor_sq = (self.X*self.X).sum()/self.Ne
        
        N_p = (self.X_p*self.X).sum()
        N_pp = (self.X_p*self.X_p).sum()
        tau = self.dt**2/self.mu
        p2 = N_p/(tau*self.Ne)
        q = (N_pp - self.Ne)/(tau**2*self.Ne)
        lambda_1 = -p2 - np.sqrt(p2**2 - q)
        lambda_2 = -p2 + np.sqrt(p2**2 - q)
        #print(f'N_p = {N_p}, N_pp = {N_pp}')
        return(lambda_1, lambda_2)
        
    def rescale(self, dens, num_gpt):
        """
        multiply density with dV per voxel
        """
        
        dV = self.V/num_gpt
        density_vox = dens*dV
        return(density_vox)
    
    def optimize(self, nsteps, density_file = None, overwrite = False):
        for i in range(nsteps):
            
            # create path to density file
            density_file = os.path.join(self.workdir, f'density_{i}')
                
            # calculate gradient for density file
            self.calculate_dEdX(density_file)
            
            # read gradient and density into python
            grad_file = os.path.join(self.workdir, 'dEdX')
            grad = self.read_gradient(grad_file)
            num_gpt_grad = len(grad)
            self.dEdX = self.rescale(grad, num_gpt_grad)
            dens = self.read_density(density_file)
            num_gpt_dens = len(dens)
            self.density = self.rescale(dens, num_gpt_dens)
            self.X = np.sqrt(self.density)
            
            # read energy
            self.energies.append(pio.parse_out_file(os.path.join(self.workdir, 'job.out'), 'TOTAL ENERGY'))

            # propagate density
            a = -self.dEdX/self.mu
            self.X_p = a*self.dt**2 + self.X
            
            # enforce that number of electrons is conserved
            lambda_1, lambda_2 = self.calculate_lambda()
            self.X_p = self.X_p + self.dt**2/self.mu*lambda_2*self.X
            
            # write new density to file
            density_p = np.power(self.X_p,2)
            
            dV = self.V/num_gpt_dens
            
            density_p = density_p/dV
            if overwrite:
                self.save_density(density_p, os.path.join(self.workdir, f'density'))
            else:
                self.save_density(density_p, os.path.join(self.workdir, f'density_{i+1}'))
            
            # update sqrt of density for previous step
            self.X_m = self.X
            #self.X = self.X_p            
    
#     def read_density(self, dens_file):
#         with open(dens_file, 'r') as f:
#             file = f.readlines()
#             density_unparsed = file[0][108:]
#             density = density_unparsed.split()
#             density = np.array([float(i) for i in density])
#             #density = density.reshape((16,16,16))
#             return(density)
        
    def read_density(self, dens_file):
        with open(dens_file, 'r') as f:
            file = f.readlines()
            raw_data = file[0].split()
            meta_data = raw_data[0:10]
            density = raw_data[10:]

            self.dens_x = meta_data[1]
            self.dens_y = meta_data[3]
            self.dens_z = meta_data[5]
            self.num_spin = meta_data[-1]

            density = np.array([float(i) for i in density])
            return(density)

    def read_gradient(self, filep):
        # load potential
        dEdX = []
        with open(filep, 'r') as f:
            for i, line in enumerate(f):
                if i != 0:
                    dEdX.append(float(line.strip('\n')))
        dEdX = np.array(dEdX)
        return(dEdX)
    
    def run_profess(self):
        os.chdir(self.workdir)
        p = subprocess.run([self.profess_path, 'job'], capture_output = True,  text=True )
        return(p)
    
    def save_density(self, density, density_path):
        dens_str = f'  x-dimension:          {self.dens_x}   y-dimension:          {self.dens_y}   z-dimension:          {self.dens_z}   # of spins:            {self.num_spin} '
        for d in density:
            dens_str += "{:.20E} ".format(d)

        new_dens = density_path
        with open(new_dens, 'w') as f:
            f.write(dens_str)
                    

class DensityOptimizerCPMD(DensityOptimizer):
      
    def initialize(self, atoms, dt, inpt_name, mu, profess_path, workdir, debug=False):
        self.workdir = workdir
        self.inpt_name = inpt_name
        self.profess_path = profess_path # path to PROFESS executable
        
        self.fs2atu = 1/(au._aut*1e15) # converts time from fs to a.t.u.: 1 fs * fs2atu = time in a.t.u.
        self.dt = dt/au.fs * self.fs2atu # conversion back to fs from ase time format by dividing by au.fs then conversion from fs to a.t.u.
        self.mu = mu
        self.energies = []
        
        self.atoms = atoms
        self.V = self.atoms.get_volume()/au.Bohr**3 # convert volume from Ang^-3 to Bohr^-3
        self.Ne = - atoms.get_initial_charges().sum() + get_num_val_elec(atoms.get_atomic_numbers())
        
        self.X = None
        self.X_m = None
        
        self.ekin_dens = [0.0] # the initial velocity of the electron density is zero for now
        
        # in CPMD coupled to ASE the density propagation is done for only one step during each iteration
        # the variale self.step counts the actual number of steps in the MD simulation
        self.step = 0
        
        self.debug = debug
        if self.debug:
            self.store_new_densities = []
            self.store_densities = []
            self.store_dEdX = []
            self.store_lambdas = []
        
    def calculate_dEdX(self, density_file):
        """
        calculates dEdX for the density specified in density file
        """
        if self.debug:
            # copy density file to work_dir/density
            shutil.copy(density_file, os.path.join(self.workdir, 'density'))
        # run PROFESS-1
        self.p = self.run_profess()
        assert self.p.returncode == 0, 'Calculation of forces failed'
    
    def calculate_lambda(self):
        #scaling_factor_sq = (self.X*self.X).sum()/self.Ne
        
        N_p = (self.X_p*self.X).sum()
        N_pp = (self.X_p*self.X_p).sum()
        tau = self.dt**2/self.mu
        p2 = N_p/(tau*self.Ne)
        q = (N_pp - self.Ne)/(tau**2*self.Ne)
        lambda_1 = -p2 - np.sqrt(p2**2 - q)
        lambda_2 = -p2 + np.sqrt(p2**2 - q)
        #print(f'N_p = {N_p}, N_pp = {N_pp}')
        return(lambda_1, lambda_2)
    
    def optimize(self, nsteps, density_file = None, overwrite = False):
        for i in range(nsteps):
            
            # do the calculation
            # improvement: find a more elegant way to get the correct density? maybe mv instead of copy
            # create path to density file
            density_file = os.path.join(self.workdir, f'density_{i}')    
            # calculate gradient for density file
            self.calculate_dEdX(density_file)
            
            # read gradient and density into python
            grad_file = os.path.join(self.workdir, 'dEdX')
            grad = self.read_gradient(grad_file)
            num_gpt_grad = len(grad)
            self.dEdX = grad # self.rescale(grad, num_gpt_grad)
            if self.X is None:
                dens = self.read_density(density_file)
                num_gpt_dens = len(dens)
                self.density = dens # self.rescale(dens, num_gpt_dens)
                self.X = np.sqrt(self.density)
            
            # read energy
            self.energies.append(pio.parse_out_file(os.path.join(self.workdir, 'job.out'), 'TOTAL ENERGY'))
            
            # propagate density
            a = -self.dEdX/self.mu
            self.X_p = a*self.dt**2 + self.X
            
            # enforce that number of electrons is conserved
            lambda_1, lambda_2 = self.calculate_lambda()
            self.X_p = self.X_p + self.dt**2/self.mu*lambda_2*self.X
            
            # write new density to file
            density_p = np.power(self.X_p,2)
            
            dV = self.V/num_gpt_dens
            
            density_p = density_p/dV
            if overwrite:
                self.save_density(density_p, os.path.join(self.workdir, f'density'))
            else:
                self.save_density(density_p, os.path.join(self.workdir, f'density_{i+1}'))
            
            # update sqrt of density
            self.X = self.X_p

    def optimize_vv(self, density_file = None):
        
        if self.debug:
            # save density every step
            density_file = os.path.join(self.workdir, f'density_{self.step}')  
        else:
            density_file = os.path.join(self.workdir, 'density') 

        # calculate gradient for density file
        self.calculate_dEdX(density_file)

        # read gradient 
        grad_file = os.path.join(self.workdir, 'dEdX')
        grad = self.read_gradient(grad_file)
        num_gpt_grad = len(grad)
        self.dEdX = self.rescale(grad, num_gpt_grad)
        
        # read density for zeroth step
        if self.X is None:
            dens = self.read_density(density_file)
            self.num_gpt_dens = len(dens)
            self.density = self.rescale(dens, self.num_gpt_dens)
            self.X = np.sqrt(self.density)

        # read energy
        self.energies.append(pio.parse_out_file(os.path.join(self.workdir, f'{self.inpt_name}.out'), 'TOTAL ENERGY'))

        # propagate density
        if self.X_m is not None:
            self.vv_step()
        else: 
            a = -self.dEdX/self.mu
            self.X_p = a*self.dt**2 + self.X

        # enforce that number of electrons is conserved
        lambda_1, lambda_2 = self.calculate_lambda()
        self.X_p = self.X_p + self.dt**2/self.mu*lambda_2*self.X

        # calculate kinetic energy of the electron density (useful for diagnostics)
        if self.X_m is None:
            v_dens = (self.X_p - self.X)/self.dt
        else:
            v_dens = (self.X_p - self.X_m)/(2*self.dt)
            
        self.ekin_dens.append( self.mu*np.sum( np.power(v_dens,2) ) )
        
        # debugging
#         print('X=', np.sum(np.power(self.X,2)))
#         if self.X_m is not None:
#             print('X_m=', np.sum(np.power(self.X_m,2)))
#         print('X_p=', np.sum(np.power(self.X_p,2)))
        
        # write new density to file
        density_p = np.power(self.X_p,2)

        dV = self.V/self.num_gpt_dens

        density_p = density_p/dV
        
        if self.debug:
            self.save_density(density_p, os.path.join(self.workdir, f'density_{self.step+1}'))
        else:
            self.save_density(density_p, os.path.join(self.workdir, f'density'))

        # update sqrt of density for previous step
        self.X_m = self.X
        self.X = self.X_p

        # update step counter
        self.step += 1
        
        if self.debug:
            self.store_densities.append(self.density)
            self.store_new_densities.append(density_p)
            self.store_dEdX.append(self.dEdX)
            self.store_lambdas.append((lambda_1, lambda_2))
            
    def run_profess(self):
        os.chdir(self.workdir)
        p = subprocess.run([self.profess_path, self.inpt_name], capture_output = True,  text=True )
        return(p)
    
    def save_property(self, property = 'ekin_dens'):
        if property == 'ekin_dens':
            fn = os.path.join(self.workdir, 'ekin_dens.txt')
            np.savetxt(fn, np.array(self.ekin_dens).T, header = 'Kinetic Energy of electron density (Ha)')

    
    def vv_step(self):
        self.X_p = 2*self.X - self.X_m - (self.dt**2/self.mu)*self.dEdX    