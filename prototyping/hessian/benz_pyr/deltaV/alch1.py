import numpy as np 

dmB=np.load('dmB.npy')


dV1=np.load('dV1.npy')
dV2=np.load('dV2.npy')
dV3=np.load('dV3.npy')

print(np.sum(np.matmul(dmB,dV1).trace() ))
print(np.sum(np.matmul(dmB,dV2).trace() ))
print(np.sum(np.matmul(dmB,dV3).trace() ))
