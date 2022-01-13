import numpy as np
from scipy import stats
from pydantic import BaseModel, Field
from typing import Optional, Tuple

def to_normal(x, loc=0, scale=1, bins = None,hist=None,e=1e-10):
    
    x = np.atleast_1d(x)
    if bins is None:
        bins = x.shape[0]
    
    if hist is None:
        hist = np.histogram(x, bins=bins)
    dist = stats.rv_histogram(hist)
    
    cdf = dist.cdf(x)
    cdf[cdf==0] = 1e-10
    cdf[cdf==1] = 1-1e-10
    return stats.norm.ppf(cdf, loc=loc, scale=scale)

class NScaler(BaseModel):
    loc: float = Field(0.)
    scale: float = Field(1.)
    bins: int = Field(10)
    hist: Optional[Tuple[np.ndarray, np.ndarray]] = Field(None)
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {np.ndarray: lambda x: x.tolist()}
        validate_assignment = True
    
    def fit(self, x):
        self.hist = np.histogram(x, bins=self.bins)
        return self
        
    def transform(self,x):
        return to_normal(x, loc=self.loc, scale=self.scale, hist=self.hist)
    
    def inverse(self,x):
        cdf = stats.norm.cdf(x, loc=self.loc, scale=self.scale)
        return stats.rv_histogram(self.hist).ppf(cdf)
        


