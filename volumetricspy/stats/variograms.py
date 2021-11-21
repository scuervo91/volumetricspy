import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt

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
    