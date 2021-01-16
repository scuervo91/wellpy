#External Imports
import pandas as pd 
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
from scipy.interpolate import interp1d
from scipy.spatial import distance_matrix
from typing import Union
import folium
from folium.plugins import MarkerCluster, MeasureControl,MousePosition#,LocateControl
import pyvista as pv 
#Local Imports
from .mincurve import min_curve_method, Survey, vtk_survey
from .interpolate import interpolate_deviation, interpolate_position
from .projection import unit_vector, projection_1d

class Well:
    """Wells [
    Well is a python Object that tries to represent a single Oil&Gas well with all its attributes.
    
    When Working with Reservoirpy, this is the main object an user will use in order to 
    connect other modules in the library and have all information organized and easily accessible. 

    All the attributes are written with a getter and setter features in order to both, validate and update the
    information.    
    ]


    Attributes
    ----------
    name : str
        Well name. It must be present in the object to be allowed to initiate an instance of it.
        The name will be used in some methods to specify a well's attributes
    rte : float
        Rotary table elevation referenced with the sea level. It is used to estimate depths in 
        tvdss in different wells attributes like Tops, Perforations, Surveys.
    surf_coord : Union[list,Point]
        Well surface coodinates. When a list is given, either a length  of 2 or 3 is allowed. [x,y,z] z is optional
        given the rte must be specified. The user can pass a shapely.Point object 
    crs : Union[int,str]
        surf_coord coordinate system. The coordinate system is used by methods in order to estimate,
        changes in coordinate systems to map or report. When pass an integer it must represent the different
        coordinate systems described in http://epsg.io/. When pass a str it must follow the next template 'EPSG:####'
        
        Example: 
         By passing 'EPSG:4326' string or 4326 are equivalent
    """
    def __init__(self,**kwargs):
        self.name = kwargs.pop('name', None)
        self.rte = kwargs.pop('rte', 0)
        self.surf_coord = kwargs.pop('surf_coord', None)
        self.crs = kwargs.pop('crs', None)
        self.survey = kwargs.pop('survey', None)

#####################################################
############## Properties ###########################

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self,value):
        assert isinstance(value,(str,type(None))), f'{type(value)} not accepted. Name must be str'
        self._name = value

    @property
    def rte(self):
        return self._rte

    @rte.setter
    def rte(self,value):
        if value is not None:
            assert isinstance(value,(int,float)), f'{type(value)} not accepted. Name must be number'
        self._rte = value

    @property
    def surf_coord(self):
        return self._surf_coord

    @surf_coord.setter
    def surf_coord(self,value):
        if value is not None:
            assert isinstance(value,(list,Point)), f'{type(value)} not accepted. Name must be shapely.geometry.Point or list [x,y,z]'
            if isinstance(value,Point):
                self._surf_coord = value
            elif isinstance(value,list):
                assert len(value) <= 3 and len(value) >= 2
                if len(value)==3:
                    self._surf_coord = Point(value[0],value[1],value[2])
                elif len(value)==2:
                    self._surf_coord = Point(value[0],value[1])
        else:
            self._surf_coord = value


    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self,value):
        if value is not None:
            if isinstance(value,str):
                assert value.startswith('EPSG:'), 'if crs is string must starts with EPSG:. If integer must be the Coordinate system reference number EPSG http://epsg.io/'
            else:
                try:
                    value = f'EPSG:{int(value)}'
                except:
                    value = None
        self._crs = value
        
    @property
    def survey(self):
        return self._survey

    @survey.setter
    def survey(self,value):
        if value is not None:
            if isinstance(value,Survey):
                self._survey = value
            elif isinstance(value,pd.DataFrame):
                assert all(i in value.columns for i in ['md','inc','azi'])
                _survey = min_curve_method(
                    value['md'],
                    value['inc'],
                    value['azi'],
                    surface_easting=self._surf_coord.x, 
                    surface_northing=self._surf_coord.y, 
                    kbe=self._rte,
                    crs=self._crs)
                self._survey = _survey
        else:
            self._survey = value
            
    def sample_deviation(self,step:int=100):
        """sample_deviation. Sample the wells deviation (md, inc, Azi) for a given step
        
        Parameters
        ----------
        step : int, optional
            Step size for the deviation, by default 100

        Returns
        -------
        pd.DataFrame
            DataFrame containing the sampled deviation

        Raises
        ------
        ValueError
            [description]
        """
        if self._survey is not None:
            _survey = self.survey
            new_dev = interpolate_deviation(_survey.index, 
                                            _survey['inc'], 
                                            _survey['azi'], md_step=step)
        else:
            raise ValueError("No survey has been set")
        return new_dev

    def sample_position(self,step:int=100)->gpd.GeoDataFrame:
        """sample_deviation. Sample the wells deviation (tvd,easting,northing) for a given step
        
        Parameters
        ----------
        step : int, optional
            Step size for the deviation, by default 100

        Returns
        -------
        pd.DataFrame
            DataFrame containing the sampled position

        Raises
        ------
        ValueError
            [description]
        """
        if self._survey is not None:
            _survey = self.survey
            new_pos = interpolate_position(_survey['tvd'], 
                                            _survey['easting'], 
                                            _survey['northing'], 
                                            tvd_step=step)
            new_pos_gpd = gpd.GeoDataFrame(new_pos,geometry=gpd.points_from_xy(new_pos.new_easting,new_pos.new_northing),crs=self._crs)
        else:
            raise ValueError("No survey has been set")
        return new_pos_gpd
    
    def to_tvd(self,md:Union[int,float,list,np.ndarray]=None, subsea:bool=False)->np.ndarray:
        if self._survey is not None:
            _survey=self.survey
            
            if md is not None:
                md = np.atleast_1d(md)
                if subsea==True:
                    _tvdss_int = interp1d(_survey.index,_survey['tvdss'],fill_value='extrapolate')
                    return _tvdss_int(md)
                    
                else:
                    _tvd_int = interp1d(_survey.index,_survey['tvd'],fill_value='extrapolate')
                    return _tvd_int(md)

    def to_coord(self,md:Union[int,float,list,np.ndarray]=None)->Point:
        if self._survey is not None:
            r=None
            _survey=self.survey
            _northing_int = interp1d(_survey['tvd'],_survey.geometry.y,fill_value='extrapolate')
            _easting_int = interp1d(_survey['tvd'],_survey.geometry.x,fill_value='extrapolate')
            _tvd_int = interp1d(_survey.index,_survey['tvd'],fill_value='extrapolate')
            if md is not None:
                md = np.atleast_1d(md)
                _tvd = _tvd_int(md)
                _northing = _northing_int(_tvd)
                _easting = _easting_int(_tvd)
                return Point(_easting,_northing,_tvd)
            
    def well_map(self,zoom=10, map_style = 'OpenStreetMap',z_unit='ft', to_crs='EPSG:4326', tooltip=False,popup=True, ax=None):
        """
        Make a Foluim map with the selected well

        Input:
            zoom -> (int, float) Initial zoom for folium map
            map_stule -> (str) Type of map folium
        Return:
            w_map -> (folium.Map) Folium map object
        """
        _coord = gpd.GeoDataFrame()

        z_coef = 0.3048 if z_unit=='ft' else 1

        x_coord = self.surf_coord.x
        y_coord = self.surf_coord.y
        z_coord = self.surf_coord.z*z_coef if self.surf_coord.has_z==True else self.rte*z_coef
        shape = self.surf_coord
        crs = self.crs
        _w = gpd.GeoDataFrame({'x':[x_coord],'y':[y_coord],'z':[z_coord],'geometry':[shape]}, index=[self.name])
        _w.crs = crs
        _w = _w.to_crs(to_crs)
        _w['lon'] = _w['geometry'].x
        _w['lat'] = _w['geometry'].y
        _coord = _coord.append(_w)
        center = _coord[['lat','lon']].mean(axis=0)

        #make the map
        if ax is None:
            map_folium = folium.Map(
                location=(center['lat'],center['lon']),
                zoom_start=zoom,
                tiles = map_style)
        else:
            assert isinstance(ax,folium.folium.Map)
            map_folium = ax

        for i, r in _coord.iterrows():
            folium.Marker(
                [r['lat'],r['lon']],
                tooltip=f"{i}" if tooltip else None,
                popup = folium.Popup(html=f"{i}",show=True) if popup else None,
                icon=folium.Icon(icon='tint', color='green')
                ).add_to(map_folium)

        folium.LayerControl().add_to(map_folium)
        #LocateControl().add_to(map_folium)
        MeasureControl().add_to(map_folium)
        MousePosition().add_to(map_folium)

        return map_folium

    def get_vtk(self):
        """
        Get the vtk object in PyVista for the well survey
        """
    
        if self.survey is None:
            raise ValueError('The survey has not been set')
        else:
            _survey = self.survey.reset_index()
            _survey = _survey.loc[:,_survey.columns != 'geometry']
            
            surv_vtk = vtk_survey(_survey[['easting','northing','tvdss']].values)
            
            for col in _survey.iteritems():
                surv_vtk.point_arrays[col[0]] = col[1].values

        return surv_vtk                
        
class WellsGroup:
    def __init__(self,*args,**kwargs):
        _well_list = []

        if args is not None:
            for i in args:
                _well_list.extend(i)
        
        self.wells = _well_list 

    @property
    def wells(self):
        return self._wells

    @wells.setter 
    def wells(self,value):
        assert isinstance(value,list)
        if not value:
            self._wells = {}
        else:
            assert all(issubclass(type(i),Well) for i in value)
            w_dict={}
            for i in value:
                w_dict[i.name] = i
            self._wells = w_dict
            
    def add_well(self,*args):
        _add_well = []

        if args is not None:
            for i in args:
                _add_well.extend(i)

        assert all(issubclass(type(i),Well) for i in _add_well)

        _wells_dict = self.wells.copy()

        for i in _add_well:
            _wells_dict[i.name] = i
        self._wells = _wells_dict
        
    def wells_surveys(self, wells:list=None, projection1d = False, azi=90, center=None):
        """
        Get a DataFrame with the wells surveys
        Input:
            wells ->  (list, None) List of wells in the Group to show
                    If None, all wells in the group will be selected
            formations ->  (list, None) List of formation in the Group to show 
                    If None, all formations in the group will be selected
        Return:
            tops -> (gpd.GeoDataFrame) GeoDataFrame with tops indexed by well
        """    
        assert isinstance(wells,(list,type(None)))
        assert isinstance(center,(list,np.ndarray, type(None)))
        assert isinstance(azi,(int,float,np.ndarray))
        # Define which wells for the distance matrix will be shown    
        if wells is None:
            _well_list = []
            for key in self.wells:
                _well_list.append(key)
        else:
            _well_list = wells

        _wells_survey = gpd.GeoDataFrame()
        for well in _well_list:
            if self.wells[well].survey is None:
                continue
            else:
                _s = self.wells[well].survey.copy()
                _s['well'] = well 
                _s = _s.reset_index()
                _wells_survey = _wells_survey.append(gpd.GeoDataFrame(_s))

        _wells_survey.crs = self.crs
        if projection1d == True:
            _pr,c = projection_1d(_wells_survey[['easting','northing']].values, azi, center=center)
            _wells_survey['projection'] = _pr
            r=[_wells_survey,c]
        else:
            r=_wells_survey

        return r
    
    def wells_surveys_ascii(self, 
        wells:list=None, 
        factor=None, 
        cols=['easting','northing','tvdss','md'],
        float_format='{:.2f}'.format
        ):
        
        assert isinstance(wells,(list,type(None)))
        
        wells_surveys_df = self.wells_surveys(wells=wells)
             
        string = ""

        if factor is None:
            factor = np.ones(len(cols))
        else:
            factor = np.atleast_1d(factor)
            assert (factor.ndim==1) & (factor.shape[0]==len(cols))
        
        for w in wells_surveys_df['well'].unique():

            _df = wells_surveys_df.loc[wells_surveys_df['well']==w,cols] * factor
            string += f"WELLNAME: {w}\n"
            string += _df.to_string(header=False,index=False,float_format=float_format) + '\n'
        return string
    
    def wells_coordinates(self, wells:list=None, z_unit='ft', to_crs='EPSG:4326'):
        """
        Get a DataFrame with the wells surface coordinates
        Input:
            wells ->  (list, None) List of wells in the Group to show the matrix. 
                    If None, all wells in the group will be selected
        Return:
            wells_coord -> (gpd.GeoDataFrame) GeoDataFrame with wells coords
        """
        assert isinstance(wells,(list,type(None)))

        # Define which wells for the distance matrix will be shown    
        if wells is None:
            _well_list = []
            for key in self.wells:
                _well_list.append(key)
        else:
            _well_list = wells

        #Create coordinates dataframe
        _coord = gpd.GeoDataFrame()

        z_coef = 0.3048 if z_unit=='ft' else 1

        for well in _well_list:
            x_coord = self.wells[well].surf_coord.x
            y_coord = self.wells[well].surf_coord.y
            z_coord = self.wells[well].surf_coord.z*z_coef if self.wells[well].surf_coord.has_z==True else self.wells[well].rte*z_coef
            shape = self.wells[well].surf_coord
            crs = self.wells[well].crs
            _w = gpd.GeoDataFrame({'x':[x_coord],'y':[y_coord],'z':[z_coord],'geometry':[shape]}, index=[well])
            _w.crs = crs
            _w = _w.to_crs(to_crs)
            _w['lon'] = _w['geometry'].x
            _w['lat'] = _w['geometry'].y
            _coord = _coord.append(_w)

        return _coord
    
    def wells_distance(self,wells:list=None, dims:list=['x','y','z'], z_unit:str='ft'):
        """
        Calculate a distance matrix for the surface coordinates of the wells

        Input:
            wells ->  (list, None) List of wells in the Group to show the matrix. 
                    If None, all wells in the group will be selected
            z ->  (Bool, default False). Take into account the z component. Z must be in the same
                    units of x, y coord
            z_unit -> (str, default 'ft') Indicate the units of the z coord. 
                    If 'ft' the z is multiplied by 0.3028 otherwise by 1

        Return:
            dist_matrix -> (pd.DataFrame) Distance matrix with index and column of wells
        """
        
        assert isinstance(wells,(list,type(None)))

        _coord = self.wells_coordinates(wells=wells, z_unit=z_unit)

        dist_array = distance_matrix(_coord[dims].values,_coord[dims].values)
        dist_matrix = pd.DataFrame(dist_array,index=_coord.index, columns=_coord.index)

        return dist_matrix

    def wells_map(self, wells:list=None,zoom=10, map_style = 'OpenStreetMap',tooltip=True,popup=False,ax=None):
        """
        Make a Foluim map with the selected wells

        Input:
            wells ->  (list, None) List of wells in the Group to show the matrix. 
                    If None, all wells in the group will be selected
            zoom -> (int, float) Initial zoom for folium map
        Return:
            w_map -> (folium.Map) Folium map object
        """
        assert isinstance(wells,(list,type(None)))

        _coord = self.wells_coordinates(wells=wells)

        center = _coord[['lat','lon']].mean(axis=0)

        #make the map
        if ax is None:
            map_folium = folium.Map(
                location=(center['lat'],center['lon']),
                zoom_start=zoom,
                tiles = map_style)
        else:
            assert isinstance(ax,folium.folium.Map)
            map_folium = ax

        for i, r in _coord.iterrows():
            folium.Marker(
                [r['lat'],r['lon']],
                tooltip=f"{i}" if tooltip else None,
                popup = folium.Popup(html=f"{i}",show=True,max_width='50%') if popup else None,
                icon=folium.Icon(icon='tint', color='green')
                ).add_to(map_folium)

        folium.LayerControl().add_to(map_folium)
        #LocateControl().add_to(map_folium)
        MeasureControl().add_to(map_folium)
        MousePosition().add_to(map_folium)

        return map_folium
    
    def wells_surveys_map(self, wells:list=None,zoom:int=10, map_style:str = 'OpenStreetMap',tooltip:bool=True,popup:bool=False,ax=None,radius=10):
        """
        Make a Foluim map with the selected wells

        Input:
            wells ->  (list, None) List of wells in the Group to show the matrix. 
                    If None, all wells in the group will be selected
            zoom -> (int, float) Initial zoom for folium map
        Return:
            w_map -> (folium.Map) Folium map object
        """
        assert isinstance(wells,(list,type(None)))

        _coord = self.wells_surveys(wells=wells)
        _coord = _coord.to_crs('EPSG:4326')
        _coord['lon'] = _coord['geometry'].x
        _coord['lat'] = _coord['geometry'].y
        center = _coord[['lat','lon']].mean(axis=0)

        #make the map
        if ax is None:
            map_folium = folium.Map(
                location=(center['lat'],center['lon']),
                zoom_start=zoom,
                tiles = map_style)
        else:
            assert isinstance(ax,folium.folium.Map)
            map_folium = ax

        for i, r in _coord.iterrows():
            folium.Circle(
                [r['lat'],r['lon']],
                tooltip=f"{r['well']} <br>md:{r['md']} <br>tvd:{r['tvd']} <br>tvdss:{r['tvdss']} <br>inc:{r['inc']} " if tooltip else None,
                popup = folium.Popup(html=f"{r['well']} <br>md:{r['md']} <br>tvd:{r['tvd']} <br>tvdss:{r['tvdss']} <br>inc:{r['inc']} ",show=True,max_width='50%') if popup else None,
                #icon=folium.Icon(icon='circle',prefix='fa', color='green'),
                radius=radius
                ).add_to(map_folium)

        folium.LayerControl().add_to(map_folium)
        #LocateControl().add_to(map_folium)
        MeasureControl().add_to(map_folium)
        MousePosition().add_to(map_folium)

        return map_folium
    
    def wells_surveys_vtk(self, wells:list=None):
        """
        Get the vtk object in PyVista for the wells survey selected
        Input:
            wells ->  (list, None) List of wells in the Group to show. 
                    If None, all wells in the group will be selected
        Return:
            surveys -> (pv.MultiBlock) pyvista.MultiBlock object with vtk surveys
        """
        if wells is None:
            _well_list = []
            for key in self.wells:
                if self.wells[key].survey is not None:
                    _well_list.append(key)
        else:
            _well_list = wells

        data = {}
        for well in _well_list:
            data[well] = self.wells[well].get_vtk()

        survey_blocks = pv.MultiBlock(data)

        return survey_blocks