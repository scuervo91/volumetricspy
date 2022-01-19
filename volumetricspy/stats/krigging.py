from statistics import variance
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.linalg import solve 
from scipy.stats import norm
from .points import CloudPoints
from .variograms import Spherical,Exponential, Gaussian


variogram_types = Union[Spherical,Exponential, Gaussian]

class OrdinaryKrigging(BaseModel):
    variogram_model: Optional[variogram_types] = Field(None)
    known_cp: Optional[CloudPoints] = Field(None)
    unknown_cp: Optional[CloudPoints] = Field(None)
    
    class Config:
        extra = 'ignore'
        validate_assignment = True
        
    def forward(
        self, 
        v:str,
        known_cp: CloudPoints = None,
        unkwown_cp: CloudPoints = None,
    ):
        known_cp = known_cp or self.known_cp
        unknown_cp = unkwown_cp or self.unknown_cp

        if v not in known_cp.df().columns:
            raise ValueError(f'{v} not in known_cp.df().columns')

        #Known Distance matrix
        kdm = known_cp.distance_matrix()
        n_knowns = kdm.shape[1]
        
        #Known and Unknown Distance Matrix
        unknown = unknown_cp.copy(deep=True)
        kudm = known_cp.distance_matrix(unknown)
        n_unknowns = kudm.shape[1]
        
        #covariance Matrix Known points
        cmk = self.variogram_model.covariance(kdm)
        
        #covariance Matrix Unknown points and Known points
        cmku = self.variogram_model.covariance(kudm)
        
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
            variance[i] = self.variogram_model.sill - np.dot(wm[:,i],cmku[:,i])
            
        #known values
        new_values = np.dot(known_cp.df()[v].values, wm[:-1,:])
        unknown.add_field(new_values,v)
        unknown.add_field(variance,f'{v}_variance')
        
        return unknown
    
