import numpy as np
import os

###########################################################################################
#                                     Make input file                                     #
###########################################################################################

def allowed_gpts_table(Ngpt):
    """
    input optimal number of gridpoints for SCF calculation
    returns closest allowed gridsize for FFT-grid in CPMD !! IF COMPILED WITH FFTW !!
    """
    gpts = [2,   3,   4,   5,   6,   7,   8,   9,  
    10,  12,14,  15,  16,  18,  20,  21,  24,  25,  27,  28,30,  32,  35, 
    36,  40,  42,  45,  48,  49,  50,54,  56,  60,  63,  64,  70,  72,  75,
    80,  81,84,  90,  96,  98, 100, 105, 108, 112, 120, 125,126, 128, 135, 
    140, 144, 147, 150, 160, 162, 168,175, 180, 189, 192, 196, 200, 210, 216
    , 224, 225,240, 243, 245, 250, 252, 256, 270, 280, 288, 294,300, 315, 
    320, 324, 336, 343, 350, 360, 375, 378,384, 392, 400, 405, 420, 432, 441
    , 448, 450, 480,486, 490, 500, 504, 512, 525, 540, 560, 567, 576,588, 
    600, 625, 630, 640, 648, 672, 675, 686, 700,720, 729, 735, 750, 756, 768
    , 784, 800, 810, 840,864, 875, 882, 896, 900, 945, 960, 972, 980,1000,
    1008,1024,1029,1050,1080,1120,1125,1134,1152,1176,1200,1215,1225,1250,
    1260,1280,1296,1323,1344,1350,1372,1400,1440,1458,1470,1500,1512,1536,
    1568,1575,1600,1620,1680,1701,1715,1728,1750,1764,1792,1800,1875,1890,
    1920,1944,1960,2000,2016,2025,2048,2058,2100,2160,2187,2205,2240,2250,
    2268,2304,2352,2400,2401,2430,2450,2500,2520,2560,2592,2625,2646,2688,
    2700,2744,2800,2835,2880,2916,2940,3000,3024,3072,3087,3125,3136,3150,
    3200,3240,3360,3375,3402,3430,3456,3500,3528,3584,3600,3645,3675,3750,
    3780,3840,3888,3920,3969,4000,4032,4050,4096,4116,4200,4320,4374,4375,
    4410,4480,4500,4536,4608,4704,4725,4800,4802,4860,4900,5000,5040,5103,
    5120,5145,5184,5250,5292,5376,5400,5488,5600,5625,5670,5760,5832,5880,
    6000,6048,6075,6125,6144,6174,6250,6272,6300,6400,6480,6561,6615,6720,
    6750,6804,6860,6912,7000,7056,7168,7200,7203,7290,7350,7500,7560,7680,
    7776,7840,7875,7938,8000,8064,8100,8192]
    
    lower_gpt = 0
    higher_gpt = 0
    for i in range(len(gpts)):
        if Ngpt >= gpts[i] and Ngpt <= gpts[i+1]:
            if gpts[i]%2 != 0:
                lower_gpt = gpts[i-1]
            else:
                lower_gpt = gpts[i]
                
            if gpts[i+1]%2 != 0:
                higher_gpt = gpts[i+2]
            else:
                higher_gpt = gpts[i+1]
            break
    return(lower_gpt, higher_gpt)

def get_boxsize(num_ve):
    """
    calculates boxsize for given number of valence electrons
    such that electron density of UEG is the same as for 38 valence electrons in a cubic box of length 20 Ang
    """
    rho_prime = 38/20**3
    r_prime = np.power(20/350, 3)

    V = num_ve/rho_prime
    L = np.power(V, 1/3)
    return(L)

def get_gpts(num_ve):
    """
    returns number of gridpoints for density/wavefunction such that density of gridpoints is as similar as possible 
    to number of gridpoints for 38 valence electrons in a cubic box of length 20 Ang
    """
    rho_prime = 38/20**3
    r_prime = np.power(20/350, 3)

    V = num_ve/rho_prime
    
    opt_gpts = np.power(V/r_prime, 1/3)
    
    lower_gpts, higher_gpts = allowed_gpts_table(opt_gpts)
    return(lower_gpts, higher_gpts)

def get_lambda(lam_val, num_ve):
    """
    calculates lambda which is closest to lam_val and which yields an integer if multiplied by
    num_ve (lambda*num_ve) == int
    """
    scaled_ve = int(np.round(lam_val*num_ve))
    new_lambda = scaled_ve/num_ve
    return(new_lambda, scaled_ve)

def write_atom(atomsym, coordinates, pp_type='GH_PBE'):
    """
    prepare the input for one atom:
    the name of the pp is 'element_name' + idx of atom in Compound object + '_SG_LDA'
    the coordinates are read from Compund as well (must be shifted to center before)
    """
    line1 = f'*{atomsym}_{pp_type} FRAC\n'
    line2 = ' LMAX=S\n'
    line3 = ' 1\n'
    line4 = ' ' + str(coordinates[0]) + ' ' + str(coordinates[1]) + ' ' + str(coordinates[2]) + '\n'
    return( [line1, line2, line3, line4] )
    
def write_atom_section(atomsymbols, coordinates):
    """
    atomsymbols: list of element names
    coordinates: list of coordinates
    concantenates inputs for individual atoms to one list where each element is one line of the input file
    """
    atom_section = ['&ATOMS\n']
    for atsym, c in zip(atomsymbols, coordinates):
        atom = write_atom(atsym, c)
        atom_section.extend(atom)
    atom_section.append('&END')
    return(atom_section)

def write_input(atomsymbols, charge, coordinates, gpts, L, write_path, template_path='/home/misa/projects/Atomic-Energies/data/cpmd_params_template.inp', debug = False):
    """
    writes input file for molecule with specified parameters boxisze L, charge, number of gpts for mesh
    """
    param_section = write_params(L, charge, gpts, template_path)
    atom_section = write_atom_section(atomsymbols, coordinates)
    with open(write_path, 'w') as f:
        f.writelines(param_section+['\n']+atom_section)
    if debug:
        return(param_section+['\n']+atom_section)
        
def write_params(L, charge, gpts, template_path='/home/misa/projects/Atomic-Energies/data/cpmd_params_template.inp'):
    """
    add correct parameters for boxsize L, charge and gpts to template
    """
    with open(template_path, 'r') as f:
        template_params = f.readlines()

    for i, line in enumerate(template_params):
        if 'CELL ABSOLUTE' in line:
            template_params[i+1] = f'        {L} {L} {L} 0.0 0.0 0.0\n'
        elif 'CHARGE' in line:
            template_params[i+1] = f'        {charge}\n'
        elif 'MESH' in line:
            template_params[i+1] = f'    {gpts} {gpts} {gpts}\n'
    return(template_params)

###########################################################################################
#                            Make pseudopotential files                                   #
###########################################################################################

def scale_ZV(zv_line, lamb):
    zv=float(zv_line.strip('ZV ='))
    new_zv = zv*lamb
    new_zv_line = '  ZV = {}\n'.format(new_zv)
    return(new_zv_line)
    
def scale_coeffs(coeffs_line, lamb):
    parts = np.array([float(_) for _ in coeffs_line.split('#')[0].strip().split()])
    parts[1:] *= lamb
    formatstring = '%4d' + (len(parts)-1)*' %20.15f' + '   #C  C1 C2\n'
    return(formatstring % (*parts,))

def generate_pp_file(lamb, element, pp_dir='/home/misa/software/PP_LIBRARY/', pp_type='_SG_LDA'):
    name_pp = element + pp_type
    f_pp = os.path.join(pp_dir, name_pp)
    
    new_pp_file = []
    for line in open(f_pp).readlines():
        if 'ZV' in line:
            new_pp_file.append(scale_ZV(line, lamb))
            continue
        if '#C' in line:
            new_pp_file.append(scale_coeffs(line, lamb))
            continue
        new_pp_file.append(line)
    new_pp_file[len(new_pp_file)-1] = new_pp_file[len(new_pp_file)-1].rstrip('\n')
    return(new_pp_file)
    
def write_pp_files_compound(compound, lamb, calc_dir, pp_dir='/home/misa/software/PP_LIBRARY/', pp_type='_SG_LDA'):
    if type(compound) == list:
        for k in compound:
            pp_file = generate_pp_file(lamb, k, pp_dir, pp_type)
            path_file = os.path.join(calc_dir, k + pp_type)
            with open(path_file, 'w') as f:
                f.writelines(pp_file)
    else:
        for k in compound.natypes.keys():
            pp_file = generate_pp_file(lamb, k)
            path_file = os.path.join(calc_dir, k + pp_type)
            with open(path_file, 'w') as f:
                f.writelines(pp_file)



