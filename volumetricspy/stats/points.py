from pydantic import BaseModel, Field, validate_arguments
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Dict, Tuple, Optional, Union
from scipy.spatial import Voronoi

from ..utils import poly_area


class Dot(BaseModel):
    x: float
    y: Optional[float] = Field(None)
    z: Optional[float] = Field(None)
    crs: Optional[int] = Field(None)
    fields: Optional[Dict[str, Union[float,str]]] = Field(None)
    
    class Config:
        extra = 'ignore'
        validate_assignment = True
        
    def df(self, to_crs:int=None):
        dict_point = self.dict(exclude=({'fields'}))
        
        if self.fields is not None:
            dict_point.update(self.fields)
        
        df = gpd.GeoDataFrame(dict_point , index=[0], geometry = gpd.points_from_xy([self.x], [self.y]), crs=self.crs)
        
        if to_crs is not None:
            df = df.to_crs(to_crs)
        
        return df
    
    def to_shapely(self):
        c = []
        for i in [self.x, self.y, self.z]:
            if i is not None:
                c.append(i)
        
        return Point(*c)
    
    def to_numpy(self):
        c = []
        for i in [self.x, self.y, self.z]:
            if i is not None:
                c.append(i)
        
        return np.array(c)
    
    @validate_arguments
    def add_field(self, d = Dict[str, Union[float,str]]):
        if self.fields is None:
            self.field = d
        else:
            self.field.update(d)
            
class CloudPoints(BaseModel):
    points: Optional[List[Dot]] = Field(None)
    
    class Config:
        extra = 'ignore'
        validate_assignment = True
        
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_point(self,dots:Union[List[Dot], Dot]):
        
        if isinstance(dots, Dot):
            dots = [dots]
        
        if self.points is None:
            self.points = dots
        else:
            self.points = self.points + dots
            
    def to_df(self, to_crs:int=None):
        df = gpd.GeoDataFrame()
        for dot in self.points:
            df = df.append(dot.df(to_crs=to_crs))
            
        return df
    
    def to_shapely(self):
        return [dot.to_shapely() for dot in self.points]
    
    def to_numpy(self):
        return np.vstack([dot.to_numpy() for dot in self.points])
    
    def poly_declusterin(self):
        df = self.to_df().reset_index(drop=True)
        vr = Voronoi(df[['X','Y']].values)
        
        vertices = vr.vertices
        regions = vr.regions 
        areas = []
        for region in regions:
            if len(region) == 0:
                continue
            
            vertices_region = vertices[region,:]
            vertices_region = np.vstack((vertices_region, vertices_region[0,:]))
            area_region = poly_area(vertices_region[:,0], vertices_region[:,1])
            
            areas.append(area_region)
        
        df['areas'] = areas
        df['weights'] = df['areas']/df['areas'].sum()
        
        for i,r in df.iterrows():
            self.points[i].add_field(d = {'area': r['areas'], 'weight': r['weights']})
            
        
        
        
        

