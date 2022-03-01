import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.linalg import solve 
from scipy.stats import norm
from .points import CloudPoints
from .variograms import Spherical,Exponential, Gaussian
from .krigging import ordinary_krigging, OrdinaryKrigging, IndicatorOridinaryKrigging


variogram_types = Union[Spherical,Exponential, Gaussian]

class SequentialGaussianSimulation(BaseModel):
    variogram_model: Optional[variogram_types] = None
    known_cp: Optional[CloudPoints] = Field(None)
    unknown_cp: Optional[CloudPoints] = Field(None)    
    max_distance: Optional[float] = Field(None)
    seed: Optional[int] = Field(None)

    class Config:
        validate_assignment = True
        
    def forward(
        self, 
        v:str,
        known_cp:Optional[CloudPoints] = None,
        unknown_cp:Optional[CloudPoints] = None,
        max_distance: Optional[float] = None,
        seed: Optional[int] = None,     
    ):
        if known_cp is None:
            known_cp = self.known_cp
        if unknown_cp is None:
            unknown_cp = self.unknown_cp
        
        #Distance matrx between known and unknown points. Known points are the rows and unknown points are the columns
        unknown = unknown_cp.copy(deep=True)
        kudm = known_cp.distance_matrix(unknown)

        # Get the minimum point distance for each known point to the unknown points
        known_df = known_cp.df()
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
            krige_point = ordinary_krigging(v,self.variogram_model,known_points_krige,unknown_point)
            #krige_point = self.ordinary_kriging(known_points_krige, unknown_point, v)
            
            #monecarlo simulation
            mc_value = norm.rvs(
                loc=krige_point.points[0].fields[f'{v}'],
                scale=krige_point.points[0].fields[f'{v}_variance'],
                size=1
            )
            #add the value to the unknown point
            unknown.points[i].add_fields({v:mc_value})
            
        return unknown
