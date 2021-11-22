from statistics import variance
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.linalg import solve 
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
    