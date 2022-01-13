import numpy as np


def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def angle_matrix(p1:np.ndarray, p2:np.ndarray):
    
    if p1.shape[1] != p2.shape[1]:
        raise ValueError(f'Points must have the same dimension p1 {p1.shape[1]} != {p2.shape[1]}')
    
    angles = np.zeros((p1.shape[0], p2.shape[0]))
    for i,m in enumerate(p1):
        for j,n in enumerate(p2):
            dif = n-m
            
            angles[i,j] = np.arctan2(dif[1], dif[0])
            
    return np.degrees(angles)

def azimuth_matrix(p1:np.ndarray, p2:np.ndarray):
    
    if p1.shape[1] != p2.shape[1]:
        raise ValueError(f'Points must have the same dimension p1 {p1.shape[1]} != {p2.shape[1]}')
    
    angles = angle_matrix(p1, p2)
    azimuths = np.zeros_like(angles)
    azimuths[(angles>90)&(angles<=180)] = 450 - angles[(angles>90)&(angles<=180)]
    azimuths[~((angles>90)&(angles<=180))] = 90 - angles[~((angles>90)&(angles<=180))]   
    return azimuths

 
        