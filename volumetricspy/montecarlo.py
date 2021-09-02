import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from scipy import stats
from typing import Union, Callable, List

class ProbVar(BaseModel):
    name: str
    dist: str = Field('norm')
    kw : dict = Field({'loc':0,'scale':1})
    factor: float = Field(1.0)
    constant: float = Field(None)
    seed : int = Field(None)

    class Config:
        validate_assignment = True
        extra = 'forbid'

    @validator('kw')
    def check_dist_build(cls,v,values):
        if isinstance(getattr(stats,values['dist'])(**v),stats._distn_infrastructure.rv_frozen):
            return v 
        else:
            raise ValueError(f"{v} are not allowed")

    def get_instance(self):
        return getattr(stats,self.dist)(**self.kw)

    def get_sample(self, size:Union[int,tuple]=None, ppf:float=None, seed=None):
        if seed is None:
            seed = self.seed
        
        if self.constant is not None:
            return self.constant
        elif size:
            return getattr(stats,self.dist)(**self.kw).rvs(size=size,random_state=seed)*self.factor
        elif ppf is not None:
            return getattr(stats,self.dist)(**self.kw).ppf(ppf)*self.factor
        else:
            return getattr(stats,self.dist)(**self.kw).mean()*self.factor
        

class MonteCarlo(BaseModel):   
    name: str 
    func: Callable[..., np.ndarray]
    args: List[ProbVar]
    
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {np.ndarray: lambda x: x.tolist()}
        
    def get_sample(self, size:Union[int,tuple]=None, ppf:float=None, seed=None):
        list_vars = []
        
        for arg in self.args:
            var_values = arg.get_sample(size=size, ppf=ppf, seed=seed)
            list_vars.append(var_values)
            
        return self.func(*list_vars)
    
    def get_sample_df(self, size:Union[int,tuple]=None, ppf:float=None, seed=None):
        vars_df = pd.DataFrame()
        list_vars = []
        for arg in self.args:
            var_values = arg.get_sample(size=size, ppf=ppf, seed=seed)
            vars_df[arg.name] = var_values
            list_vars.append(var_values)
        
        vars_df[self.name] = self.func(*list_vars)
        if ppf is not None:
            vars_df.index = ppf
            
        return vars_df
        
    


        