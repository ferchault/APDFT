import numpy as np
import psi4
psi4.set_memory('2 GB')
psi4.core.be_quiet()
import os
scratch_dir = os.environ.get('TMPDIR')
if scratch_dir:
	psi4_io = psi4.core.IOManager.shared_object()
	psi4_io.set_default_path(scratch_dir + '/')

import scipy.spatial.transform as sst

elements = 'CHHCHHCHHCHH'
coords = np.array((-0.80582974850563949, -0.73660437274591195, 0.07486479611839635,-1.25002541558480607, -1.11272962173038881, 1.00207583520636812,-1.30963477283345409, -1.22704114377832951, -0.76427319948392614,-0.73948232221053856, 0.80621170581959434, -0.02721954739247012,-1.20426086103227648, 1.22328096314051415, -0.92640498379068181,-1.14465150378362801, 1.33759248518845486, 0.83994405089961199,0.80582974850607436, 0.73660437274574542, -0.07486479611840029,1.25002541558439595, 1.11272962173114265, -1.00207583520640320,1.30963477283304441, 1.22704114377908335, 0.76427319948389105,0.73948232221097354, -0.80621170581976120, 0.02721954739246620,1.14465150378406300, -1.33759248518862206, -0.83994405089961599,1.20426086103179153, -1.22328096314152246, 0.92640498379076364,)).reshape(-1, 3)

def build_geo(alpha_left, alpha_right, phi):
    geo = coords.copy()
       
    # alpha_left
    axis = np.cross(geo[6]-geo[3], geo[0]-geo[3])
    rot_full = sst.Rotation.from_rotvec(alpha_left * axis / np.linalg.norm(axis))
    rot_half = sst.Rotation.from_rotvec(alpha_left * axis / np.linalg.norm(axis) / 2)
    com = geo[3].copy()
    geo -= com
    geo[:3] = rot_full.apply(geo[:3])
    geo[3:6] = rot_half.apply(geo[3:6])
    geo += com
    
    # alpha_right
    axis = np.cross(geo[6]-geo[3], geo[0]-geo[3])
    rot_full = sst.Rotation.from_rotvec(-alpha_right * axis / np.linalg.norm(axis))
    rot_half = sst.Rotation.from_rotvec(-alpha_right * axis / np.linalg.norm(axis) / 2)
    com = geo[6].copy()
    geo -= com
    geo[9:] = rot_full.apply(geo[9:])
    geo[6:9] = rot_half.apply(geo[6:9])
    geo += com
    
    
    # phi
    axis = geo[3] - geo[6]
    com = geo[6].copy()
    rot = sst.Rotation.from_rotvec(phi * axis / np.linalg.norm(axis))
    geo -= com
    geo[6:] = rot.apply(geo[6:])
    geo += com
    
    return geo

def get_elements(modification):
    up_elem = list(elements[:])
    dn_elem = list(elements[:])
    for idx, letter in enumerate(modification):
        up_elem[idx*3] = letter
        if letter == 'N':
            dn_elem[idx*3] = 'B'
        else:
            dn_elem[idx*3] = 'N'
    return up_elem, dn_elem

def get_line(modification, alpha_left, alpha_right, phi):
    geo = build_geo(2*np.pi/360*alpha_left, 2*np.pi/360*alpha_right, 2*np.pi/360*phi)
    
    this_elements = get_elements(modification)
    energies = []
    for direction in range(2):
        molstr = ''
        for elem, pos in zip(this_elements[direction], geo):
            molstr += '%s %.15f %.15f %.15f\n' % (elem, *pos)
        mol = psi4.geometry(molstr)
        mol.update_geometry()
        energies.append(psi4.energy('HF/6-31G*', molecule=mol))
    print (modification, alpha_left, alpha_right, phi, energies[0] - energies[1])

import sys
for alpha in np.linspace(0, 90, 20):
	for phi in np.linspace(0, 180, 20):
		get_line('NBBN', alpha, alpha, phi)

