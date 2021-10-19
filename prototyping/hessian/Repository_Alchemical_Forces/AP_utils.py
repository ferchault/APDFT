charge2symbol={1:"H",2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne"}

def alias_param(param_name: str, param_alias: str):
    """
    Decorator for aliasing a param in a function
    Args:
        param_name: name of param in function to alias
        param_alias: alias that can be used for this param
    Returns:
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            alias_param_value = kwargs.get(param_alias)
            if param_alias in kwargs.keys():
                kwargs[param_name] = alias_param_value
                del kwargs[param_alias]
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


def printxyz(coords,al,fn):
    atomnumber={1:"H",5:"B",6:"C",7:"N"}
    assert len(al)==len(coords)
    with open(fn,"w")as xyzf:
        xyzf.write(str(len(al))+" \n" )
        xyzf.write("molecule \n" )
        for i in range(len(coords)):
            xyzf.write(atomnumber[al[i]]+"    "+str(coords[i])[1:-1]+"\n")
    return


def parse_charge(dL):
    """ There are two options: 
    1) call FcM(**kwargs,fcs=[c1,c2,--cn]) with a list of length equal to the number of atoms
    2) FcM(**kwargs,fcs=[[aidx1,aidx2,..,aidxn],[c1,c2,..cn]]) with a list of two sublist for atoms' indexes and fract charges
    """
    a=[[],[]]
    parsed=False
    if len(dL) ==2:   #necessario, ma non sufficiente per caso 2
        try:
            len(dL[0])==len(dL[1])
            if isinstance(dL[0][0],int) or isinstance(dL[0][0],float):
                parsed=True
        except: pass
    if not parsed and (isinstance(dL[0],int) or isinstance(dL[0],float)): #move to case2 
        for i in range(len(dL)):
            if dL[i]!=0:
                a[0].append(i)
                a[1].append(dL[i])
        dL=a
        parsed=True
    if not parsed:
        print("Failed to parse charges")
        raise
    return dL

def DeltaV(mol,dL):
    """dL=[[i1,i2,i3],[c1,c2,c3]]"""
    mol.set_rinv_orig_(mol.atom_coords()[dL[0][0]])
    dV=mol.intor('int1e_rinv')*dL[1][0]
    for i in range(1,len(dL[0])): 
        mol.set_rinv_orig_(mol.atom_coords()[dL[0][i]])
        dV+=mol.intor('int1e_rinv')*dL[1][i]
    return -dV