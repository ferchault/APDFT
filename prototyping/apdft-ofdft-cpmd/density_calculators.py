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
    
    def initialize(self, atoms, dt, mu, workdir):
        self.workdir = workdir
        
        self.dt = dt
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
    
    def read_density(self, dens_file):
        with open(dens_file, 'r') as f:
            file = f.readlines()
            density_unparsed = file[0][108:]
            density = density_unparsed.split()
            density = np.array([float(i) for i in density])
            #density = density.reshape((16,16,16))
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
        p = subprocess.run(['/home/misa/software/PROFESS-1/PROFESS', 'job'], capture_output = True,  text=True )
        return(p)
    
    def save_density(self, density, density_path):
        dens_str = '  x-dimension:          16   y-dimension:          16   z-dimension:          16   # of spins:            1 '
        for d in density:
            dens_str += "{:.20E} ".format(d)

        new_dens = density_path
        with open(new_dens, 'w') as f:
            f.write(dens_str)


class DensityOptimizerCPMD(DensityOptimizer):
    
    def initialize(self, atoms, dt, mu, workdir):
        self.workdir = workdir
        
        self.dt = dt
        self.mu = mu
        self.energies = []
        
        self.atoms = atoms
        self.V = self.atoms.get_volume()/au.Bohr**3
        self.Ne = - atoms.get_initial_charges().sum() + get_num_val_elec(atoms.get_atomic_numbers())
        
        self.X = None
        self.X_m = None
    
    
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
            self.dEdX = self.rescale(grad, num_gpt_grad)
            if self.X is None:
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
            
            # update sqrt of density
            self.X = self.X_p
    
    def optimize_vv(self, nsteps, density_file = None, overwrite = False):
        for i in range(nsteps):
            
            # do the calculation, find a more elegant way to get the correct density? maybe mv instead of copy
            # create path to density file
            density_file = os.path.join(self.workdir, f'density_{i}')    
            # calculate gradient for density file
            self.calculate_dEdX(density_file)
            
            # read gradient and density into python
            grad_file = os.path.join(self.workdir, 'dEdX')
            grad = self.read_gradient(grad_file)
            num_gpt_grad = len(grad)
            self.dEdX = self.rescale(grad, num_gpt_grad)
            if self.X is None:
                dens = self.read_density(density_file)
                num_gpt_dens = len(dens)
                self.density = self.rescale(dens, num_gpt_dens)
                self.X = np.sqrt(self.density)
            
            # read energy
            self.energies.append(pio.parse_out_file(os.path.join(self.workdir, 'job.out'), 'TOTAL ENERGY'))
            
            # propagate density
            if self.X_m is not None:
                self.vv_step()
            else: 
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
            self.X = self.X_p
    
    def vv_step(self):
        self.X_p = 2*self.X - self.X_m - (self.dt**2/self.mu)*self.dEdX    