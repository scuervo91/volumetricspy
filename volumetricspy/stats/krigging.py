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

def inverse_weighted_distance(
    known_points: CloudPoints,
    unknown_points: CloudPoints,
    v:str,
    p:float = 1.0,
    max_distance: Optional[float] = None
):
    
    if v not in known_points.df().columns:
        raise ValueError(f'{v} not in known_points.df().columns')
    
    unknown = unknown_points.copy(deep=True)
    
    kudm = known_points.distance_matrix(unknown_points)
    
    if max_distance is not None:
        kudm[kudm>max_distance] = 0
    
    inverse_distance = 1 / np.power(kudm,p)
    inverse_distance[np.isinf(inverse_distance)] = 0
    
    weights = inverse_distance/inverse_distance.sum(axis=0)
    
    real_values = known_points.df()[v].values.reshape(-1,1)
    
    pred_values = np.matmul(weights.T, real_values)
    
    unknown.add_field(pred_values,v)
    
    return unknown
    

def ordinary_krigging(
    known_points: CloudPoints,
    unknown_points: CloudPoints,
    v:str,
    variogram_model: Optional[variogram_types] = None,
):

    if v not in known_points.df().columns:
        raise ValueError(f'{v} not in known_points.df().columns')

    #Known Distance matrix
    kdm = known_points.distance_matrix()
    n_knowns = kdm.shape[1]
    
    #Known and Unknown Distance Matrix
    unknown = unknown_points.copy(deep=True)
    kudm = known_points.distance_matrix(unknown)
    n_unknowns = kudm.shape[1]
    
    #covariance Matrix Known points
    cmk = variogram_model.covariance(kdm)
    
    #covariance Matrix Unknown points and Known points
    cmku = variogram_model.covariance(kudm)
    
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
        variance[i] = variogram_model.sill - np.dot(wm[:,i],cmku[:,i])
        
    #known values
    new_values = np.dot(known_points.df()[v].values, wm[:-1,:])
    unknown.add_field(new_values,v)
    unknown.add_field(variance,f'{v}_variance')
    
    return unknown



class KriggingBase(BaseModel):
    known_cp: Optional[CloudPoints] = Field(None)
    unknown_cp: Optional[CloudPoints] = Field(None)    

    class Config:
        validate_assignment = True

class OrdinaryKrigging(KriggingBase):
    variogram_model: Optional[variogram_types] = None
    
    def forward(
        self,
        v:str,
        variogram_model: Optional[variogram_types] = None,
    ):
        variogram_model = variogram_model or self.variogram_model
        return ordinary_krigging(
            self.known_cp,
            self.unknown_cp,
            v,
            variogram_model,

        )
        

class IndicatorOridinaryKrigging(KriggingBase):
    variogram_model: Union[variogram_types,Dict[str,variogram_types]] = Field(None)
    
    def forward(
        self,
        v: str,
        variogram_model: Optional[Union[variogram_types,Dict[str,variogram_types]]] = None,
        argmax: bool = True,
        seed = None
    ):
        if variogram_model is None:
            variogram_model = self.variogram_model
        #variogram_model = variogram_model or self.variogram_model
        #convert the indicator variable to one hot encoding
        kn = self.known_cp.one_hot_encode(v)
        
        # List of categorical variables
        cats = kn.df()[v].unique().tolist()
        ukn = self.unknown_cp.copy()
        for i in cats:
            ukn = ordinary_krigging(
                kn,
                ukn,
                i,
                variogram_model=variogram_model[i] if isinstance(variogram_model,dict) else variogram_model,
            )
            
        if argmax:
            pred = ukn.df()[cats].idxmax(axis=1).to_frame()
            pred.columns=[v]
            ukn.add_fields_from_df(pred, [v])
        else:
            rng = np.random.default_rng(seed=seed)
            preds = []
            for i, r in ukn.df().iterrows():
                weights = r[cats].values.astype(float)
                probs = np.exp(weights)/np.exp(weights).sum()
                pred = rng.choice(cats, p=probs)
                preds.append(pred)
            ukn.add_field(preds, v)
            
        return ukn
            
        
class IWD(KriggingBase):
    p: float = Field(1.0, ge=1.0)
    max_distance: Optional[float] = Field(None, gt=0.0)
    
    def forward(
        self,
        v:str,
        p: Optional[float] = None,
        max_distance: Optional[float] = None
    ):
        max_distance = max_distance or self.max_distance
        p = p or self.p
        return inverse_weighted_distance(
            self.known_cp,
            self.unknown_cp,
            v,
            p,
            max_distance
        )