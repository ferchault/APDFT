import sys
def read_xyz(fn):
    lines = open(fn).readlines()[2:]
    elements = []
    coords = []
    for line in lines:
        parts = line.strip().split()
        elements.append('d')
        coords.append([float(_) for _ in parts[1:]])
    return elements, coords

znuc = sys.argv[1].split('-')
elements, coords = read_xyz('benzene.xyz')
lookup = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 5: 'B'}
print ('12\n')
for z, coord in zip(znuc, coords):
	print (lookup[int(z)], *coord)
