import matplotlib.pyplot as plt
import sys

PathToQM9XYZ = '/home/simon/QM9/XYZ/'

if __name__ == "__main__":
    f = open(PathToQM9XYZ+str(sys.argv[1])+'.xyz', "r")
    data = f.read()
    f.close()
    print(data)
    xdata = []
    ydata = []
    zdata = []
    labels = []
    N = int(data.splitlines(False)[0])
    for i in range(2,N+2): #get the atoms one by one
        line = data.splitlines(False)[i]
        labels.append(line[0])
        x = line.split('\t')
        count = 0
        while count < len(x):
            if x[count] == ' ' or x[count] == '':
                del x[count]
            else:
                count += 1
        xdata.append(float(x[1]))
        ydata.append(float(x[2]))
        zdata.append(float(x[3]))

    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata)
    for j in range(len(xdata)):
        ax.text(xdata[j], ydata[j], zdata[j], labels[j])
    plt.show()
