import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
from skimage import measure
from scipy.integrate import simps
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from zmapio import ZMAPGrid
from pydantic import BaseModel, Field, validator, validate_arguments
from typing import Dict, Tuple, List, Union
import folium
from folium.plugins import MeasureControl,MousePosition
def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

class Surface(BaseModel):
    name: str = Field(...)
    shape: Tuple[int,int] = Field(None)
    x: np.ndarray = Field(None)
    y: np.ndarray = Field(None)
    z: np.ndarray = Field(None)
    crs: int = Field(None)
    fields: Dict[str,np.ndarray] = Field(None)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {np.ndarray: lambda x: x.tolist()}
        validate_assignment = True
        
       
    @validator('x')
    def check_shape(cls,v,values):
        length = values['shape'][1]
        assert v.ndim == 1
        assert v.shape[0] == length, f'Shape mismatch: {v.shape} != {length}'
        return v

    @validator('y')
    def check_shape(cls,v,values):
        length = values['shape'][0]
        assert v.ndim == 1
        assert v.shape[0] == length, f'Shape mismatch: {v.shape} != {length}'
        return v

    @validator('z')
    def check_shape(cls,v,values):
        length = values['shape'][0] * values['shape'][1]
        assert v.ndim == 1
        assert v.shape[0] == length, f'Shape mismatch: {v.shape} != {length}'
        return v

    @validator('fields')
    def check_shape(cls,v,values):
        length = values['shape'][0] * values['shape'][1]
        for i in v:
            assert v[i].ndim == 1
            assert v[i].shape[0] == length, f'Shape mismatch: {v[i].shape} != {length}'
        return v
    
    def from_z_map(self,value, factor_z = -1, crs=4326):

        z_file = ZMAPGrid(value)
        z_df = z_file.to_dataframe().dropna()
        z_df['Z'] *= factor_z
        p = z_df.pivot(index='Y',columns='X',values='Z')
        p.sort_index(axis=0, inplace=True)
        p.sort_index(axis=1, inplace=True)

        self.x = np.array(p.columns)
        self.y = np.array(p.index)
        self.z = p.values.flatten()
        self.shape = p.values.shape
        self.crs=crs
    
    def get_mesh(self):
        xx, yy = np.meshgrid(self.x, self.y)
        zz = self.z.reshape(self.shape)
        return xx, yy, zz

    def contour(self,ax=None,**kwargs):

        #Create the Axex
        cax= ax or plt.gca()
        xx, yy, zz = self.get_mesh()
        return cax.contour(xx,yy,zz,**kwargs)

    def contourf(self,ax=None,**kwargs):

        #Create the Axex
        cax= ax or plt.gca()
        xx, yy, zz = self.get_mesh()
        return cax.contourf(xx,yy,zz,**kwargs)

    def structured_surface_vtk(self):

        #Get a Pyvista Object StructedGrid
        xx, yy, zz = self.get_mesh()
        grid = pv.StructuredGrid(xx, yy, zz).elevation()

        return grid
    
    def get_contours(self,levels=None,zmin=None,zmax=None,n=10):
        
        #define levels
        if levels is not None:
            assert isinstance(levels,(np.ndarray,list))
            levels = np.atleast_1d(levels)
            assert levels.ndim==1
        else:
            zmin = zmin if zmin is not None else np.nanmin(self.z)
            zmax = zmax if zmax is not None else np.nanmax(self.z)

            levels = np.linspace(zmin,zmax,n)

        _,_,zz = self.get_mesh()
        xmax = np.nanmax(self.x)
        ymax = np.nanmax(self.y)
        xmin = np.nanmin(self.x)
        ymin = np.nanmin(self.y)

        #iterate over levels levels
        data = pd.DataFrame()
        i = 0
        for level in levels:
            contours = measure.find_contours(zz,level)

            if contours == []:
                continue
            else:
                for contour in contours:
                    level_df = pd.DataFrame(contour, columns=['y','x'])
                    level_df['z'] = level
                    level_df['n'] = i
                    data = data.append(level_df,ignore_index=True)
                    i += 1

        if not data.empty:
            #re scale
            data['x'] = (data['x']/zz.shape[1]) * (xmax - xmin) + xmin
            data['y'] = (data['y']/zz.shape[0]) * (ymax - ymin) + ymin

        return data
        
    def get_contours_bound(self,levels=None,zmin=None,zmax=None,n=10):
        #define levels
        if levels is not None:
            assert isinstance(levels,(np.ndarray,list))
            levels = np.atleast_1d(levels)
            assert levels.ndim==1
        else:
            zmin = zmin if zmin is not None else np.nanmin(self.z)
            zmax = zmax if zmax is not None else np.nanmax(self.z)

            levels = np.linspace(zmin,zmax,n)

        xmax = np.nanmax(self.x)
        ymax = np.nanmax(self.y)
        xmin = np.nanmin(self.x)
        ymin = np.nanmin(self.y)

        #iterate over levels levels
        contours = self.structured_surface_vtk().contour(isosurfaces=levels.tolist())

        contours.points[:,2] = contours['Elevation']

        df = pd.DataFrame(contours.points, columns=['x','y','z'])

        #Organize the points according their angle with respect the centroid. This is done with the 
        #porpuse of plot the bounds continously.
        list_df_sorted = []
        for i in df['z'].unique():

            df_z = df.loc[df['z']==i,['x','y','z']]
            centroid = df_z[['x','y']].mean(axis=0).values
            df_z[['delta_x','delta_y']] = df_z[['x','y']] - centroid
            df_z['angle'] = np.arctan2(df_z['delta_y'],df_z['delta_x'])
            df_z.sort_values(by='angle', inplace=True)


            list_df_sorted.append(df_z)


        return pd.concat(list_df_sorted, axis=0)


    def get_contours_area_bounds(self,levels=None,n=10,zmin=None,zmax=None,c=2.4697887e-4):

        contours = self.get_contours_bound(levels=levels,zmin=zmin,zmax=zmax,n=n)


        area_dict= {}
        for i in contours['z'].unique():
            poly = contours.loc[contours['z']==i,['x','y']]
            area = poly_area(poly['x'],poly['y'])
            area_dict.update({i:area*c})

        return pd.DataFrame.from_dict(area_dict, orient='index', columns=['area'])

    def get_contours_area_mesh(self,levels=None,n=10,zmin=None,zmax=None,c=2.4697887e-4):


        zmin = zmin if zmin is not None else np.nanmin(self.z)
        zmax = zmax if zmax is not None else np.nanmax(self.z)

        if levels is not None:
            assert isinstance(levels,(np.ndarray,list))
            levels = np.atleast_1d(levels)
            assert levels.ndim==1
        else:
            levels = np.linspace(zmin,zmax,n)
        xx, yy, zz = self.get_mesh()
        dif_x = np.diff(xx,axis=1).mean(axis=0)
        dif_y = np.diff(yy,axis=0).mean(axis=1)
        dxx, dyy = np.meshgrid(dif_x,dif_y) 
        
        area_dict = {}
        for i in levels:
            z = zz.copy()
            z[(z<i)|(z>zmax)|(z<zmin)] = np.nan
            z = z[1:,1:]
            a = dxx * dyy * ~np.isnan(z) *2.4697887e-4
            area_dict.update({i:a.sum()})

        return pd.DataFrame.from_dict(area_dict, orient='index', columns=['area'])



    def get_contours_gdf(self,levels=None,zmin=None,zmax=None,n=10, crs=None):
        
        #define levels
        if levels is not None:
            assert isinstance(levels,(np.ndarray,list))
            levels = np.atleast_1d(levels)
            assert levels.ndim==1
        else:
            zmin = zmin if zmin is not None else np.nanmin(self.z)
            zmax = zmax if zmax is not None else np.nanmax(self.z)

            levels = np.linspace(zmin,zmax,n)

        xx, yy, zz = self.get_mesh()
        
        xmax = np.nanmax(self.x)
        ymax = np.nanmax(self.y)
        xmin = np.nanmin(self.x)
        ymin = np.nanmin(self.y)

        #iterate over levels levels
        data = gpd.GeoDataFrame()
        i = 0
        for level in levels:
            poly_list =[]
            contours = measure.find_contours(zz,level)

            if contours == []:
                continue
            else:
                for contour in contours:
                    level_df = pd.DataFrame(contour, columns=['y','x'])

                    #Re scale
                    level_df['x'] = (level_df['x']/zz.shape[1]) * (xmax - xmin) + xmin
                    level_df['y'] = (level_df['y']/zz.shape[0]) * (ymax - ymin) + ymin

                    #List of tuples
                    records = level_df[['x','y']].to_records(index=False)
                    list_records = list(records)

                    if len(list_records)<3:
                        continue
                    else:
                        poly = Polygon(list(records))

                    #Append to list of Polygon
                    poly_list.append(poly)

            # Make Multipolygon
            multi_poly = MultiPolygon(poly_list)

            #Make a Geo dataframe
            level_gdf = gpd.GeoDataFrame({'level':[level],'geometry':[multi_poly]})

            # Append data to general geodataframe
            data = data.append(level_gdf,ignore_index=True)
            i += 1
        
        #Add data crs
        data.crs = self.crs 
        
        #Convert to defined crs
        if crs is not None:
            data = data.to_crs(crs)
        return data

    def surface_map(
        self, 
        levels=None,
        zmin=None,
        zmax=None,
        n=10, 
        crs=4326,
        zoom=10, 
        map_style = 'OpenStreetMap', 
        ax=None,
        fill_color='OrRd', 
        fill_opacity=1, 
        line_opacity=1,
    ):
    
        gdf = self.get_contours_gdf(levels=levels,zmin=zmin,zmax=zmax,n=n,crs=crs).reset_index()

        
        if ax is None:
            centroid_gdf = gdf.to_crs(self.crs).centroid.to_crs(crs)
            centroid_df = pd.DataFrame({'lon':centroid_gdf.x,'lat':centroid_gdf.y})
            center = centroid_df[['lat','lon']].mean(axis=0)
            ax = folium.Map(
                location=(center['lat'],center['lon']),
                zoom_start=zoom,
                tiles = map_style)
        
        folium.Choropleth(
            geo_data=gdf.to_json(),
            data=gdf[['index','level']],
            columns=['index','level'],
            key_on='feature.properties.index',
            fill_color=fill_color, 
            fill_opacity=fill_opacity, 
            line_opacity=line_opacity,
            legend_name='Level [ft]',
        ).add_to(ax)
        
        folium.LayerControl().add_to(ax)
        #LocateControl().add_to(map_folium)
        MeasureControl().add_to(ax)
        MousePosition().add_to(ax)
        
        return ax

    def get_contours_area(self,levels=None,n=10, group=True,c=2.4697887e-4):

        #c is the conversion factor from m2 to acre

        #get contours
        contours = self.get_contours(levels=levels,n=n)

        if contours.empty:
            print('None contours found')
            return pd.Series(np.zeros(levels.shape[0]), index=levels, name='area')
        #dataframe
        data = pd.DataFrame()

        for level in contours['level'].unique():
            level_df = contours.loc[contours['level']==level,:]

            for n in level_df['n'].unique():
                poly_df = level_df.loc[level_df['n']==n,:]

                area = poly_area(poly_df['x'].values, poly_df['y'].values) * c

                area_df = pd.DataFrame({'level':[level],'n':[n],'area':area})

                data = data.append(area_df,ignore_index=True)

        if group:
            data_g = data[['level','area']].groupby('level').sum()
            return data_g
        else:
            return data

    def get_volume(self,levels=None, n=10,c=2.4697887e-4):
        
        area = self.get_contours_area(levels=levels,n=n,c=c,group=True)

        #Integrate
        rv=simps(area['area'],np.abs(area.index))

        return rv, area

    def get_z(self, x, y, method='linear'):
        xx, yy, zz = self.get_mesh()
        _x = xx.flatten()
        _y = yy.flatten()
        _z = zz.flatten()

        _xf = _x[~np.isnan(_z)]
        _yf = _y[~np.isnan(_z)]
        _zf = _z[~np.isnan(_z)] 

        return griddata((_xf,_yf),_zf,(x,y), method=method)
    


class SurfaceGroup(BaseModel):
    name:str = Field(None)
    surfaces: Dict[str,Surface] = Field(None)
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
    
    @validator('surfaces')
    def check_epsg(cls,v):
        list_epsg = [v[i].crs for i in v]
        assert all(x == list_epsg[0] for x in list_epsg)
        return v

    @validate_arguments
    def add_surface(self,surf:Union[Surface,List[Surface]]):
        list_surfaces = []
        if isinstance(surf,Surface):
            list_surfaces.append(surf)
        else:
            list_surfaces.extend(surf)
        
        surf_dict = {i.name:i for i in list_surfaces}
        
        if self.surfaces is None:
            self.surfaces = surf_dict
        else:
            self.surfaces.update(surf_dict)
            
    @validate_arguments  
    def create_parallel_surfaces(
        self,
        base_surf:str,
        thickness:Union[float,List[float]],
    ):
        list_tickness = []
        if isinstance(thickness,list):
            list_tickness.extend(thickness)
        else:
            list_tickness.append(thickness)
        
        list_surfaces = []
        for i,thick in enumerate(thickness):
            surf_copy = self.surfaces[base_surf].copy()
            surf_copy.z = surf_copy.z + thick
            surf_copy.name = f'{base_surf}_{thick}'
            list_surfaces.append(surf_copy)
        
        self.add_surface(list_surfaces)
            
    def get_volume_bounds(self, 
        top_surface=None, 
        bottom_surface=None, 
        levels=None, 
        zmin=None,
        zmax=None,
        n=20,c=2.4697887e-4,method='mesh'):

        assert all([top_surface is not None,bottom_surface is not None])

        #define levels
        if levels is not None:
            assert isinstance(levels,(np.ndarray,list))
            levels = np.atleast_1d(levels)
            assert levels.ndim==1
        else:
            zmin = zmin if zmin is not None else np.nanmin(self.surfaces[bottom_surface].z)
            zmax = zmax if zmax is not None else np.nanmax(self.surfaces[top_surface].z)

            levels = np.linspace(zmin,zmax,n)

        if method=='mesh':
            top_area = self.surfaces[top_surface].get_contours_area_mesh(levels=levels,n=n,c=c,zmin=zmin, zmax=zmax)
            bottom_area = self.surfaces[bottom_surface].get_contours_area_mesh(levels=levels,n=n,c=c,zmin=zmin, zmax=zmax)

        elif method=='contours':
            top_area = self.surfaces[top_surface].get_contours_area_bounds(levels=levels,n=n,c=c,zmin=zmin, zmax=zmax)
            bottom_area = self.surfaces[bottom_surface].get_contours_area_bounds(levels=levels,n=n,c=c, zmin=zmin, zmax=zmax)

        #Merge two contours ara for top and bottom indexed by depth
        area=top_area.merge(bottom_area,how='outer',left_index=True,right_index=True,suffixes=['_top','_bottom']).fillna(0)
        area['dif_area']= np.abs(area['area_top'] - area['area_bottom'])
        area['height'] = np.diff(area.index, append=0)
        area['vol'] = area['dif_area'].multiply(area['height'])
        rv = area['vol'].iloc[0:-1].sum()
        #area['height'] = area.index-area.index.min()
        #area['tick']=np.diff(area['height'], prepend=0)
        #area['vol'] = area['dif_area'] * area['tick']
        #Integrate
        #rv=simps(area['dif'],area['thick'])
        #rv=area['vol'].sum()

        return rv, area.iloc[0:-1]


    def get_volume(self, 
        top_surface=None, 
        bottom_surface=None, 
        levels=None, 
        zmin=None,
        zmax=None,
        n=20,c=2.4697887e-4):

        assert all([top_surface is not None,bottom_surface is not None])

        #define levels
        if levels is not None:
            assert isinstance(levels,(np.ndarray,list))
            levels = np.atleast_1d(levels)
            assert levels.ndim==1
        else:
            zmin = zmin if zmin is not None else np.nanmin(self.surfaces[bottom_surface].z)
            zmax = zmax if zmax is not None else np.nanmax(self.surfaces[top_surface].z)

            levels = np.linspace(zmin,zmax,n)

        top_area = self.surfaces[top_surface].get_contours_area(levels=levels,n=n,c=c,group=True)
        bottom_area = self.surfaces[bottom_surface].get_contours_area(levels=levels,n=n,c=c,group=True)

        #Merge two contours ara for top and bottom indexed by depth
        area=top_area.merge(bottom_area,how='outer',left_index=True,right_index=True,suffixes=['_top','_bottom']).fillna(0)
        area['dif_area']= np.abs(area['area_top'] - area['area_bottom'])
        area['height'] = area.index-area.index.min()
        area['tick']=np.diff(area['height'], prepend=0)
        area['vol'] = area['dif_area'] * area['tick']
        #Integrate
        #rv=simps(area['dif'],area['thick'])
        rv=area['vol'].sum()

        return rv, area

    def structured_surface_vtk(self, surfaces:list=None):
        
        if surfaces is None:
            _surface_list = []
            for key in self.surfaces:
                _surface_list.append(key)
        else:
            _surface_list = surfaces

        data={}
        for s in _surface_list:
        #Get a Pyvista Object StructedGrid
            data[s] = self.surfaces[s].structured_surface_vtk()

        grid_blocks = pv.MultiBlock(data)

        return grid_blocks