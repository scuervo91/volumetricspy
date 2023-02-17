import numpy as np
import pandas as pd

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

 
distance_converter_dict = {
    'from':['m','ft','m','ft'],
    'to': ['ft','m','m','ft'],
    'value':[3.28,0.3048,1,1]
}
distance_matrix_converter = pd.DataFrame(distance_converter_dict).pivot(index='from',columns='to', values='value')

area_converter_dict = {
    'from':['m2','ft2','m2','ft2','acre','acre','acre','ft2','m2'],
    'to': ['ft2','m2','m2','ft2','acre','ft2','m2','acre','acre'],
    'value':[10.7639,0.092903,1,1,1,43560,4046.86,1/43560,1/4046.86]
}
area_matrix_converter = pd.DataFrame(area_converter_dict).pivot(index='from',columns='to', values='value')

volume_converter_dict = {
    'from':[
        'bbl','bbl','bbl','bbl','bbl','bbl',
        'ft3','ft3','ft3','ft3','ft3','ft3',
        'ft3','bbl','m3','m3','m3','m3','m3',
        'm3','m3','Mbbl','Mbbl','Mbbl','Mbbl',
        'Mbbl','Mbbl','Mbbl','MMbbl','MMbbl','MMbbl',
        'MMbbl','MMbbl','MMbbl','MMbbl','Mft3','Mft3',
        'Mft3','Mft3','Mft3','Mft3','Mft3','MMft3','MMft3',
        'MMft3','MMft3','MMft3','MMft3','MMft3'
    ],
    'to': [
        'ft3','m3','Mbbl','MMbbl','Mft3','MMft3',
        'bbl','Mbbl','MMbbl','Mft3','MMft3','m3',
        'ft3','bbl','m3','bbl','ft3','Mft3','MMft3',
        'Mbbl','MMbbl','Mbbl','bbl','MMbbl','MMft3',
        'Mft3','ft3','m3','MMbbl','Mbbl','MMft3','Mft3',
        'bbl','ft3','m3','Mft3','MMft3','ft3','MMbbl','Mbbl',
        'bbl','m3','Mft3','MMft3','ft3','MMbbl','Mbbl',
        'bbl','m3'
    ],
    'value':[
        5.615,1/6.28981,1e-3,1e-6,5.615e-3,5.615e-6,
        1/5.615,1e-3/5.615,1e-6/5.615,1e-3,1e-6,
        0.0283168,1,1,1,6.28981,35.3147,35.3147e-3,35.3147e-6,
        6.28981e-3,6.28981e-6,1,1e3,1e-3,5.615e-3,5.615,5.615e3,1e3/6.28981,
        1,1e3,5.615,5.615e3,1e-6,5.615e6,1e6/6.28981,1,1e-3,1e3,1e-3/5.615,
        1/5.615,1e3/5.615,1e3/35.2875,1e3,1,1e6,1/5.615,1e3/5.615,1e6/5.615,1e6/35.2875
    ]
}
volume_matrix_converter = pd.DataFrame(volume_converter_dict).pivot(index='from',columns='to', values='value')