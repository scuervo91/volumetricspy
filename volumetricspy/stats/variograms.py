from statistics import variance
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.linalg import solve 
from scipy.stats import norm
from .points import CloudPoints

class Variogram(BaseModel):
    sill: float = Field(1.)
    nugget: float = Field(0.)
    range: Optional[float] = Field(None)
    
    def plot(self,h, ax=None, **kwargs):
        ax = ax or plt.gca()
        gamma = self.forward(h)
        ax.plot(h,gamma, **kwargs)
        return ax
    
    def covariance(self,h):
        return self.sill - self.forward(h)
    
    def ordinary_kriging(
        self,
        known:CloudPoints, 
        unknown:CloudPoints, 
        v:str,
        **kwargs
    ):  
        
        #Known Distance matrix
        kdm = known.distance_matrix()
        n_knowns = kdm.shape[1]
        
        #Known and Unknown Distance Matrix
        kudm = known.distance_matrix(unknown)
        n_unknowns = kudm.shape[1]
        
        #covariance Matrix Known points
        cmk = self.covariance(kdm)
        
        #covariance Matrix Unknown points and Known points
        cmku = self.covariance(kudm)
        
        #put matrix with constrains
        cmk = np.vstack((cmk, np.ones(n_knowns)))
        cmk = np.column_stack((cmk, np.ones(n_knowns+1)))
        cmk[-1,-1] = 0
        
        cmku = np.vstack((cmku, np.ones(n_unknowns)))
        
        #weights matrix
        wm = np.zeros((n_knowns+1,n_unknowns))
        variance =np.zeros(n_unknowns)
        
        for i in range(n_unknowns):
            wm[:,i] = solve(cmk,cmku[:,i])
            variance[i] = self.sill - np.dot(wm[:,i],cmku[:,i])
            
        #known values
        new_values = np.dot(known.df()[v].values, wm[:-1,:])
        unknown.add_field(new_values,v)
        unknown.add_field(variance,f'{v}_variance')
        
        return unknown
            
    def sgs(
        self, 
        known:CloudPoints,
        unknown:CloudPoints,
        v:str,
        max_distance: float = None,
        seed: int = None,
    ):

        #Distance matrx between known and unknown points. Known points are the rows and unknown points are the columns
        kudm = known.distance_matrix(unknown)
        
        # Get the minimum point distance for each known point to the unknown points
        known_df = known.df()
        known_df['unkown_idx'] = np.argmin(kudm, axis=1)
        known_df = known_df[~known_df.duplicated(subset=['unkown_idx'])]
        
        #assing the value of variable of interest to the unknown points based on the minimum distance removing duplicates
        for i,r in known_df.iterrows():
            unknown.points[r['unkown_idx']].add_fields({v:r[v]})
            
        unknowns_df = unknown.df()
        n_total = unknowns_df.shape[0]
        unknowns_df = unknowns_df[unknowns_df[v].isnull()].sample(frac=1, random_state=seed)
        n_unknowns = unknowns_df.shape[0]
        
        print(f'{n_total} total points, {n_unknowns} unknown points')
        
        for i,r in unknowns_df.iterrows():

            #extract the unknown point to be interpolated
            unknown_point = unknown.subset(i)
            
            #extract the known points that are close to the unknown point
            knowns_df = unknown.df()
            knowns_df = knowns_df[knowns_df[v].notna()]
            if max_distance is not None:
                unknowns_cp = unknown.subset(knowns_df.index)
                
                distances = unknown_point.distance_matrix(unknowns_cp)
                knowns_df['distance'] = np.squeeze(distances)
                knowns_df = knowns_df[knowns_df['distance'] <= max_distance]
                
                idx_krige = knowns_df.index
            else:
                idx_krige = knowns_df.index
            
            #extract the known points to use when krigging
            known_points_krige = unknown.subset(idx_krige)
            #make krigging

            krige_point = self.ordinary_kriging(known_points_krige, unknown_point, v)
            
            #monecarlo simulation
            mc_value = norm.rvs(
                loc=krige_point.points[0].fields[f'{v}'],
                scale=krige_point.points[0].fields[f'{v}_variance'],
                size=1
            )
            #add the value to the unknown point
            unknown.points[i].add_fields({v:mc_value})
            
        return unknown
            
            
        
        
        
        

        
        
        
        
        
class Spherical(Variogram):
    
    def forward(self, h):
        h = np.atleast_1d(h)
        gamma = np.zeros_like(h)
        
        gamma[h>=self.range] = self.sill
        
        h_fil = h[h<self.range]
        gamma[h<self.range] = self.sill * (1.5*(h_fil/self.range) - 0.5*np.power(h_fil/self.range,3))
        
        return gamma + self.nugget
    
class Exponential(Variogram):
    
    def forward(self, h):
        h = np.atleast_1d(h)
        return self.sill * (1.0 - np.exp(-h/self.range)) + self.nugget
    
class Gaussian(Variogram):
    
    def forward(self, h):
        h = np.atleast_1d(h)
        return self.sill * (1 - np.exp(-np.square(h/self.range))) + self.nugget
    