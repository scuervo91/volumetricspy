from pydantic import BaseModel, Field, validate_arguments
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Dict, Tuple, Optional, Union
from scipy.spatial import Voronoi, voronoi_plot_2d, distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import poly_area, azimuth_matrix, angle_matrix


class Dot(BaseModel):
    x: float
    y: Optional[float] = Field(None)
    z: Optional[float] = Field(None)
    crs: Optional[int] = Field(None)
    fields: Optional[Dict[str, Union[float,str]]] = Field(None)
    
    class Config:
        extra = 'ignore'
        validate_assignment = True
        
    def gdf(self,to_crs:int=None):
        dict_point = self.dict(exclude=({'fields'}))
        
        if self.fields is not None:
            dict_point.update(self.fields)
        
        df = gpd.GeoDataFrame(dict_point , index=[0], geometry = gpd.points_from_xy([self.x], [self.y]), crs=self.crs)
        
        if to_crs is not None:
            df = df.to_crs(to_crs)
                
        return df
    
    def df(self):
        dict_point = self.dict(exclude=({'fields'}))
        
        if self.fields is not None:
            dict_point.update(self.fields)
        
        df = pd.DataFrame(dict_point , index=[0])
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
    def add_fields(self, d = Dict[str, Union[float,str]]):
        if self.fields is None:
            self.fields = d
        else:
            self.fields.update(d)
            
    def add_field(self, key, value):
        if self.fields is None:
            self.fields = {key:value}
        else:
            self.fields[key] = value
            
class CloudPoints(BaseModel):
    points: Optional[List[Dot]] = Field(None)
    
    class Config:
        extra = 'ignore'
        validate_assignment = True
    
    def npoints(self):
        return len(self.points)
    
    def sample(self, n:int):
        return CloudPoints(points=np.random.choice(self.points, n, replace=False).tolist())
    
    def subset(self, idx:Union[int, List[int]]):
        if isinstance(idx, int):
            idx = [idx]
        
        p = [self.points[i] for i in idx]
            
        return CloudPoints(points=p)
    
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_point(self,dots:Union[List[Dot], Dot]):
        
        if isinstance(dots, Dot):
            dots = [dots]
        
        if self.points is None:
            self.points = dots
        else:
            self.points = self.points + dots

    def df(self):
        df_list = []
        for dot in self.points:
            df_list.append(dot.df())
        
        return pd.concat(df_list, axis=0).reset_index(drop=True)

    def gdf(self,to_crs:int=None):
        gdf_list = []
        for dot in self.points:
            gdf_list.append(dot.gdf(to_crs=to_crs))
            
        return gpd.GeoDataFrame(pd.concat(gdf_list, axis=0).reset_index(drop=True))
    
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def from_df(
        self, 
        df:pd.DataFrame,
        x:str='x',
        y:Optional[str]=None,
        z:Optional[str]=None, 
        crs:Optional[int]=None, 
        fields:Optional[Union[str, List[str]]]=None
    ):
        
        for i,r in df.iterrows():
            _dot = Dot(
                x = r[x],
                y = r[y] if y is not None else None,
                z = r[z] if z is not None else None,
                fields = r[fields].to_dict() if fields is not None else None,
                crs = crs
            )
            self.add_point(_dot)
            
        return self
    
    def add_fields_from_df(self, df:pd.DataFrame, fields:List[str]):
        for i,p in enumerate(self.points):
            for f in fields:
                p.add_field(f, df[f].iloc[i])
        
        return self
    
    def add_field(self, field:Union[List[float], np.ndarray], name:str):
        for i,p in enumerate(self.points):
            p.add_field(name, field[i])
            
        return self

    
    def to_shapely(self):
        return [dot.to_shapely() for dot in self.points]
    
    def to_numpy(self):
        return np.vstack([dot.to_numpy() for dot in self.points])
    
    def veronoi(self):
        df = self.df().reset_index(drop=True)
        return Voronoi(df[['x','y']].values)
    
    def plot_veronoi(self, ax=None, **kwargs):
        ax = ax or plt.gca()    
        vr = self.veronoi()
        fig = voronoi_plot_2d(vr, ax=ax, **kwargs)
        
    def azimuth_matrix(self, other=None):
        p = self.to_numpy()
        
        if other is None:
            return azimuth_matrix(p, p)
        return azimuth_matrix(p, other.to_numpy())
        
    def distance_matrix(self, other=None):
        p = self.to_numpy()
        
        if other is None:
            return distance_matrix(p, p)
        return distance_matrix(p, other.to_numpy())
        
    
    def plot(self, hue:str=None, ax=None, **kwargs):
        ax = ax or plt.gca()
        df = self.df()
        
        return sns.scatterplot(data=df, x='x', y='y', hue=hue, ax=ax, **kwargs)
    
    def plot_mesh(self, v:str,ax=None,**kwargs):
        ax = ax or plt.gca()
        df = self.df().pivot(index='y', columns='x', values=v)
        xx, yy = np.meshgrid(df.columns, df.index)
        
        d = ax.pcolormesh(xx,yy,df.values, **kwargs)
        plt.colorbar(d)
        
        return ax       
    
    def variogram(
        self, 
        var:str,
        lag_dist:float=None,
        lag_tol:float=None,
        nlags:int = 10,
        tmin=None, 
        tmax=None,
        azi: float = None,
        azi_tol:float = 20,
        bandwidth:float = None
    ):
        #number points
        npoints = self.npoints()
         
        coords = self.df()
        #Calculate distance MATRIX and put in a dataframe  
        distances = pd.DataFrame(
            self.distance_matrix(),
            index = pd.Index(range(npoints), name='i'),
            columns=pd.Index(range(npoints), name='j')
        ).stack().reset_index()
        distances.columns = ['i', 'j', 'distance']
        
        #calculate azimuths
        azimuths = pd.DataFrame(
            self.azimuth_matrix(),
            index = pd.Index(range(npoints), name='i'),
            columns=pd.Index(range(npoints), name='j')
        ).stack().reset_index()
        azimuths.columns = ['i', 'j', 'azimuth']
        
        #Merge the two dataframes
        df = distances.merge(azimuths, how='inner', left_on=['i','j'], right_on=['i','j'])
        df = df.merge(coords, how='left', left_on='i', right_index=True)
        df = df.merge(coords, how='left', left_on='j', right_index=True,suffixes=("_i","_j"))
        
        
        #delete points with distance equal to zero and other filters
        df = df[df['distance']>0]
        
        if tmin is not None:
            df = df[df['distance']>=tmin]

        if tmax is not None:
            df = df[df['distance']<=tmax]
        
        #Make lag distance array
        if lag_dist is None:
            lag_dist = df['distance'].max() / nlags
        if lag_tol is None:
            lag_tol = lag_dist * 0.5
        
        init_edge = lag_dist - lag_tol
        final_edge = lag_dist*nlags + lag_tol
        lags = np.linspace(init_edge,final_edge,nlags+1)
        
        df['lag'] = pd.to_numeric(pd.cut(df['distance'], lags, labels=lags[:-1]))
        
        
        if azi is not None:
            
            azi_min = azi - azi_tol
            azi_max = azi + azi_tol
            
            df = df[(df['azimuth']>=azi_min)&(df['azimuth']<=azi_max)]
            
        if bandwidth is not None:
            if azi >270:
                alpha = 450 - azi
            else:
                alpha = 90 - azi
                
            alpha_rad = np.deg2rad(alpha)
            x1, y1  = np.cos(alpha_rad-np.pi*0.5)*bandwidth*0.5, np.sin(alpha_rad-np.pi*0.5)*bandwidth*0.5
            x2, y2  = np.cos(alpha_rad+np.pi*0.5)*bandwidth*0.5, np.sin(alpha_rad+np.pi*0.5)*bandwidth*0.5
            
            x_m, y_m = np.cos(alpha_rad), np.sin(alpha_rad)
            m = y_m / x_m
            
            df['m'] = m
            
            df['b1'] = -m*(x1+df['x_i']) + (y1+df['y_i'])
            df['b2'] = -m*(x2+df['x_i']) + (y2+df['y_i'])
            
            df['p'] = -m*(df['x_j']) + (df['y_j'])
            df['fil_1'] = np.sign(df['b1'] - df['p'])
            df['fil_2'] = np.sign(df['b2'] - df['p'])
            df['fil_prod'] = df['fil_1']*df['fil_2']
            
            df = df[df['fil_prod']<0] 
            
        df['var'] = np.square(df[f'{var}_i'] - df[f'{var}_j'])*0.5
        df_gr = df.groupby('lag')['var'].mean().reset_index()
        
            
        return df.reset_index(drop=True), df_gr
            
            
        

        
        
        
        
        
        
        
        

       
    # def poly_declustering(self):
       
    #     vr = self.veronoi()
        
    #     vertices = vr.vertices
    #     regions = vr.regions 
    #     areas = []
    #     for region in regions:
    #         if len(region) == 0:
    #             continue
            
    #         vertices_region = vertices[region,:]
    #         vertices_region = np.vstack((vertices_region, vertices_region[0,:]))
    #         area_region = poly_area(vertices_region[:,0], vertices_region[:,1])
            
    #         areas.append(area_region)
            
    #     areas_arr = np.array(areas)
    #     weight_ar = areas_arr / np.sum(areas_arr)
                
    #     for i,v in enumerate(zip(areas_arr, weight_ar)):
    #         self.points[i].add_field(d = {'area': v[0], 'weight': v[1]})
            
    #     return self
            
        
        
        
        

