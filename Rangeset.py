import pandas as pd
import numpy as np

from shapely.ops import triangulate
from shapely.geometry import MultiPoint, MultiLineString, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union

from bokeh.palettes import Category10, Category20

class Rangeset():
    '''Create a rangeset instance.
    
    Parameters
    ----------
    pp: numpy.array
        2D positions of the embedded data points.
    df: pandas.DataFrame
        DataFrame from which the embedding was generated with one row per data point
        (may contain additional attributes not used for the embedding).
    '''
    def __init__(self, pp, df):
        self.threshold = 1
        self.size_inside = 3
        self.size_outside = 8
        self.colormap_default = ['#2b83ba', '#afcc50', '#ffe05c', '#f5a60a', '#d7191c']
        self.labels_default = {'#2b83ba': 'very low', 
                       '#afcc50': 'low', 
                       '#ffe05c': 'medium', 
                       '#f5a60a': 'high', 
                       '#d7191c': 'very high'}
        
        self.colormap = ['#2b83ba', '#afcc50', '#ffe05c', '#f5a60a', '#d7191c']
        self.labels = {'#2b83ba': 'very low', 
                       '#afcc50': 'low', 
                       '#ffe05c': 'medium', 
                       '#f5a60a': 'high', 
                       '#d7191c': 'very high'}
        
        self.points_2d = pp
        self.df = df.reset_index()
    
    ###############################################################################
    
    def color2label(self, c):
        '''Return the variable name for a given color.
        '''
        if c in self.labels:
            return self.labels[c]
        return 'unknown'
    
    ###############################################################################
    
    def _get_histogram_bounds(self, var, val_range=None, bins=5):
        '''Compute histogram edges for a given variable and range.
        '''
        # bounds for the histogram
        if val_range == None:
            val_range = (self.df[var].min(), self.df[var].max())
        step = (val_range[1]-val_range[0])/bins
        bounds = [val_range[0]+i*step for i in range(bins+1)]
        bounds[-1] = bounds[-1]+0.0001

        # include all points in the filtering
        bounds_full = list(bounds)
        bounds_full[0]  = self.df[var].min()
        bounds_full[-1] = self.df[var].max() + 0.0001
        
        return bounds, bounds_full
    
    ###############################################################################
    
    def _max_edge(self, polygon):
        '''Compute the length of the longest edge of a polygon.
        
        Parameters
        ----------
        polygon: shapely.Polygon
            The polygon to be tested.
            
        Returns
        -------
        max_edge: float
            Length of the longest edge.
        '''
        return max([LineString([p,q]).length for p,q in zip(polygon.boundary.coords[:-1],polygon.boundary.coords[1:])])
    
    ###############################################################################
    
    def _update_filters(self, var, val_range, bins):
        '''Determine a matching colormap for a given variable.
        
        Parameters
        ----------
        var: str
            The variable used for the rangeset.
        val_range: tuple of floats (min,max)
            Custom (min,max)-range used as boundaries for the discretization.
        bins: int
            Number of bins for the discretization.
            
        Returns
        -------
        bounds: array
            Histogram edges for (min,max)-limited histogram.
        bounds_full: array
            Histogram edges covering the full range of data.
        '''
        vals = sorted(self.df[var].unique())
        if len(vals) < 15:
            bins = len(vals)
            val_range=None
            if bins == 2:
                # https://medium.com/sketch-app-sources/colors-to-use-in-a-dashboard-babf030d44d5
                self.colormap = ['#5da5da','#b2912f']
            else:
                cmap = ['#5da5da','#faa43a','#60bd68','#f17cb0','#b2912f','#b276b2','#decf3f','#f15854']
                cmap = ['#5da5da','#f17cb0','#b2912f','#b276b2','#decf3f','#f15854','#faa43a','#60bd68']
                self.colormap = cmap[:bins] if bins <= 8 else Category20[bins]
            self.labels = {c: str(i) for c,i in zip(self.colormap,range(1,bins+1))}
        else:
            self.colormap = self.colormap_default
            self.labels = self.labels_default
            
        return self._get_histogram_bounds(var, val_range, bins=bins)
    
    ###############################################################################
    
    def compute_contours(self, var, val_range=None, bins=5):
        '''Compute the rangeset contours for a given variable.
        
        Parameters
        ----------
        var: string
            A variable contained in the DataFrame.
        val_range: tuple of floats (min,max)
            Custom (min,max)-range used as boundaries for the discretization.
        bins: int
            Number of bins for the discretization.
            
        Returns
        -------
        polygons: pandas.DataFrame
            DataFrame matching the multi_polygons data format of bokeh. Represents the boundary contours of the rangeset.
        points: pandas.DataFrame
            DataFrame matching the scatter data format of bokeh. Contains colored scatter data with respective scaling 
            for inliers and outliers.
        bounds: array
            Boundaries used during discretization.
        cnt_in: array of int
            Number of inlying points per bin.
        cnt_out: array of int
            Number of outlying points per bin.
        '''
        bounds, bounds_full = self._update_filters(var, val_range, bins)
        
        # return values
        cnts = []
        cnt_in = []
        cnt_out = []

        scatter_pos = [[],[]]
        scatter_size = []
        scatter_color = []

        poly_pos = [[],[]]
        poly_color = []
        
        for v_min,v_max,c in zip(bounds_full[:-1], bounds_full[1:], self.colormap):
            df_select = self.df[(self.df[var] >= v_min) & (self.df[var]< v_max)]
            points    = MultiPoint(self.points_2d[df_select.index])
            poly      = unary_union([polygon for polygon in triangulate(points) if self._max_edge(polygon) < self.threshold])
            
            filter_points = points
            
            # points are in one or multiple polygons
            if not poly.is_empty:
                # convert single polygons to multipolygon
                poly = MultiPolygon([poly]) if isinstance(poly, Polygon) else poly
                
                for p in poly.geoms:
                    # ensure uniform boundary representation
                    boundary = MultiLineString([p.boundary]) if isinstance(p.boundary, LineString) else p.boundary
                    
                    bb = boundary[0].coords.xy
                    holes_x = [h.coords.xy[0].tolist() for h in boundary[1:]]
                    holes_y = [h.coords.xy[1].tolist() for h in boundary[1:]]
                    poly_pos[0].append([[bb[0].tolist()]+holes_x])
                    poly_pos[1].append([[bb[1].tolist()]+holes_y]) 
                    
                    poly_color.append(c)
            
                # store inlier points
                filter_points = list(filter(poly.intersects, points)) 
                cnt_in.append(len(filter_points))
                coords = np.array([pp.coords.xy for pp in filter_points]).reshape(len(filter_points),2).T
                scatter_pos = np.append(scatter_pos, coords, 1)
                scatter_color += [c]*len(coords[0])
                scatter_size += [self.size_inside]*len(coords[0])
                
                # store outlier points
                filter_points = list(filter(poly.disjoint, points)) 
            else:
                cnt_in.append(0)
                
            cnt_out.append(len(filter_points))
            coords = np.array([pp.coords.xy for pp in filter_points]).reshape(len(filter_points),2).T
            scatter_pos = np.append(scatter_pos, coords, 1)
            scatter_color += [c]*len(coords[0])
            scatter_size += [self.size_outside]*len(coords[0])
                    
        return (pd.DataFrame({'xs': poly_pos[0], 'ys': poly_pos[1], 'color': poly_color}),
                pd.DataFrame({'x': scatter_pos[0], 'y': scatter_pos[1], 'size': scatter_size, 'color': scatter_color}),
                bounds,
                cnt_in,
                cnt_out)
    
