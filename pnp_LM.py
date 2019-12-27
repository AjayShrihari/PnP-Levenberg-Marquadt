import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import sys
import os
import numpy as np
import numpy.linalg as LA
from copy import deepcopy
import random
#from scipy.sparse.linalg import svds, eigs

def read(txtfile):
    line=txtfile.readline()
    # print(line)
    line=line.split()
    groundC=list(map(float,line))
    groundC=np.vstack((np.array(groundC).reshape(3,4)))
    #print(groundC)
    return groundC
#file = open("poses.txt",'r')
K=np.array([7.070912e+02, 0.000000e+00, 6.018873e+02, 0.000000e+00, 7.070912e+02, 1.831104e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape(3,3)
Porig = np.array([-9.098548e-01,5.445376e-02, -4.113381e-01, -1.872835e+02, 4.117828e-02, 9.983072e-01, 4.107410e-02, 1.870218e+00, 4.128785e-01, 2.043327e-02, -9.105569e-01, 5.417085e+01]).reshape(3,4)
P=deepcopy(Porig)
#print (P.shape)
pcd = o3d.io.read_point_cloud("pcl.ply")
xyz_load = np.asarray(pcd.points)
#print (xyz_load.shape)
#print (xyz_load.shape)
ones = np.ones((xyz_load.shape[0],1))
#print (xyz_load.shape[0])
xyz_load = np.concatenate((xyz_load,ones),axis = 1 )
#print (xyz_load.shape)
point2d = []
point2d = np.matmul(P,xyz_load.T)
point2d = point2d/point2d[2,:] 
point2d = point2d[0:2,:].T
#print (point2d.shape)

trans = []
ind=random.sample(range(0,len(point2d)),1000)
for i in ind:
    x = xyz_load[i,0]
    y = xyz_load[i,1]
    z = xyz_load[i,2]
    u = point2d[i,0]
    v = point2d[i,1]
    trans.append([-x,-y,-z,-1,0,0,0,0,u*x,u*y,u*z,u])
    trans.append([0,0,0,0,-x,-y,-z,-1,v*x,v*y,v*z,v])

trans = np.array(trans).astype('float32')
#print(trans.shape)
U,D,V = LA.svd(trans)
#print (trans)
P = V[-1:]
P = P.reshape(3,4)
P=P*Porig[2,3]/P[2,3]
print (P)
print(Porig)

points3d=np.array([]).reshape(0,4)

for i in range(1100):
    x = xyz_load[i,0]
    y = xyz_load[i,1]
    z = xyz_load[i,2]
    points3d=np.vstack((points3d,np.array([x,y,z,1])))
    
#print(points3d.shape)
errchange = np.array([1000]*12)
Ptemp=(P.flatten()).reshape(12,1)
prevP=(P.flatten()).reshape(12,1)
orig2dpoints=point2d[:1100,:]
grad=100
#print(orig2dpoints.shape)
iter=0
err = 1000
preverr=10000
while preverr-err>0:
    prevP=deepcopy(Ptemp)
    preverr=err
    # calculate jacobian
    print('iteration: ',iter)
    iter=iter+1
    J=np.array([]).reshape(0,12)
    points2d=(Ptemp.reshape((3,4))).dot(points3d.T)
    points2d=points2d/points2d[2,:]
    for i in range(len(points3d)):
        J=np.vstack((J,np.vstack((K[0,0]*np.hstack(( ((points3d[i,:].T)).reshape((1,4)), np.array([0,0,0,0]).reshape(1,4) , -(points2d[0,i])*((points3d[i,:].T).reshape((1,4))) )),K[1,1]*np.hstack(( np.array([0,0,0,0]).reshape(1,4), ((points3d[i,:].T)).reshape((1,4)) , -(points2d[1,i])*((points3d[i,:].T).reshape((1,4))) ))))))
    
    error=points2d[:2,:]-orig2dpoints.T
    error=error.flatten().reshape(2*error.shape[1],1)
    grad=LA.norm((J.T).dot(error))
    err=sum(sum(abs(Porig-Ptemp.reshape(3,4))))
    Ptemp=Ptemp-0.1*(LA.pinv((J.T).dot(J)).dot(J.T)).dot(error)
    print('grad: ',grad)
    print('error: ',err)
    print(Ptemp.reshape(3,4))
    print(Porig)

print(Ptemp.reshape(3,4))