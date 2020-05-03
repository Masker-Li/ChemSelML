# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:03:00 2020

@author: Licheng Xu
"""

import numpy as np

def Test(test_radius,pointnumber,AN,x,y,z,atom_rad):
    phi = 2*np.pi * np.random.rand(pointnumber,1)
    costheta = -1 + 2*np.random.rand(pointnumber,1)
    u = 1*np.random.rand(pointnumber,1)
    
    theta = np.arccos(costheta)
    r = test_radius * pow(u,1/3)
    
    test_x = r * np.sin(theta) * np.cos(phi)
    test_y = r * np.sin(theta) * np.sin(phi)
    test_z = r * np.cos(theta)
    test_xyz = np.array([test_x,test_y,test_z]).T.reshape(-1,3)
    xyz = np.array([x,y,z]).T
    dist = np.reshape(np.sum(xyz**2,axis=1),(xyz.shape[0],1))+ np.sum(test_xyz**2,axis=1)-2*xyz.dot(test_xyz.T)
    atom_rad = np.array(atom_rad).reshape(-1,1)
    count = dist - atom_rad**2 <= 0
    #count = np.zeros((pointnumber,1))
    #count = np.array([1 if (test_x[n][0] - x[t])**2+(test_y[n][0]-y[t])**2 + (test_z[n][0]-z[t])**2 <= atom_rad[t]**2 else 0 for n in range(pointnumber) for t in range(AN)]).reshape(-1,1)
    
    #for n in range(pointnumber):
    #    for t in range(AN):
    #        if (test_x[n][0] - x[t])**2+(test_y[n][0]-y[t])**2 + (test_z[n][0]-z[t])**2 <= atom_rad[t]**2:
    #            count[n][0] = 1  
    #            break
    
    percent = np.average(count.any(axis=0))
    return percent

#%%


def BV_engine(coordinate,Atom_label):
    
    Atom_label -= 1
    atom2rad_dict = {1:1.2, 2:1.4, 3:1.82, 4:1.53, 5:1.92, 6:1.7, 7:1.55 , 8:1.52,
                     9:1.47, 10:1.54, 11:2.27, 12:1.73, 13:1.84, 14:2.1, 15:1.8, 16:1.8,
                     17:1.75, 18:1.88, 19:2.75, 20:2.31, 35:1.85}
    
    
    AN = coordinate.shape[0]
    x = coordinate[:,1]
    y = coordinate[:,2]
    z = coordinate[:,3]
    
    #进行分子坐标平移
    move_x = coordinate[Atom_label][1]
    move_y = coordinate[Atom_label][2]
    move_z = coordinate[Atom_label][3]
    x = x - move_x
    y = y - move_y
    z = z - move_z
    xyz_new = np.array([x,y,z])
    xyz_new = xyz_new.T    #不确定是不是要这一步
    #最终分子的xyz坐标
    
    atom = list(coordinate[:,0].astype(int))
    atom_rad = [atom2rad_dict[A] for A in atom]
    
    pointnumber = 500000    #定义总测试点数目
    n_test = 10             #test次数
    
    test_radius_3 = 3       #定义测试球体的半径
    percent_list_3A = np.array([Test(test_radius_3,pointnumber,AN,x,y,z,atom_rad) for j in range(n_test)])

    mean_percent_3A = np.average(percent_list_3A)
    var_percent_3A = np.var(percent_list_3A)
    
    test_radius_4 = 4       #定义测试球体的半径
    percent_list_4A = np.array([Test(test_radius_4,pointnumber,AN,x,y,z,atom_rad) for j in range(n_test)])

    mean_percent_4A = np.average(percent_list_4A)
    var_percent_4A = np.var(percent_list_4A)
    
    test_radius_5 = 5       #定义测试球体的半径
    percent_list_5A = np.array([Test(test_radius_5,pointnumber,AN,x,y,z,atom_rad) for j in range(n_test)])

    mean_percent_5A = np.average(percent_list_5A)
    var_percent_5A = np.var(percent_list_5A)

    return mean_percent_3A,mean_percent_4A,mean_percent_5A,var_percent_3A,var_percent_4A,var_percent_5A





















