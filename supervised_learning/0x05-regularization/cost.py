import numpy as np

with np.load("arrays.npz") as data:

    thrLayer = data['thrLayer'] # The final layer post activation; you
    # can derive this final layer, if verification needed, using weights below

    thetaO = data['thetaO'] # The weight array between layers 1 and 2
    thetaT = data['thetaT'] # The weight array between layers 2 and 3

    Ynew = data['Ynew'] # The output array with a 1 in position i and 0s elsewhere

    #class i is the class that the data described by X[i,:] belongs to

    X = data['X'] #Raw data with 1s appended to the first column
    Y = data['Y'] #One dimensional column vector; entry i contains the class of entry i


m = len(thrLayer)
k = thrLayer.shape[1]
cost = 0

Y_arr = np.zeros(Ynew.shape)
for i in xrange(m):
    Y_arr[i,int(Y[i,0])-1] = 1

for i in range(m):
    for j in range(k):
        cost += -Y_arr[i,j]*np.log(thrLayer[i,j]) - (1 - Y_arr[i,j])*np.log(1 - thrLayer[i,j])
cost /= m

'''
Regularized Cost Component
'''

regCost = 0

for i in range(len(thetaO)):
    for j in range(1,len(thetaO[0])):
        regCost += thetaO[i,j]**2

for i in range(len(thetaT)):
    for j in range(1,len(thetaT[0])):
        regCost += thetaT[i,j]**2
lam=1
regCost *= lam/(2.*m)


print(cost)
print(cost + regCost)