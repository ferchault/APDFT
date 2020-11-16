def read_input(file):
    """
    read input parameters for pyscf calculation
    """
    input_parameters = {'basis':None, 'intg_meth':None, 'structure_file':None, 'lambda':None}
    with open(file, 'r') as f:
        for line in f:
            keyword = line.split()[0]
            if keyword in input_parameters.keys():
                input_parameters[keyword] = line.split()[1]
                
    return(input_parameters)