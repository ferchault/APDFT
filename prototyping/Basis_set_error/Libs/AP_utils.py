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