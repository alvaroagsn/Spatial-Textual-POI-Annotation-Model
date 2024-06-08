import pandas as pd
import geopandas as gpd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from shapely import wkt
import networkx as nx

from sklearn.preprocessing import LabelEncoder
import torch

import numpy as np
import pickle as pkl
import os

from libpysal import weights
from libpysal.cg import voronoi_frames

class Util:
    def __init__(self) -> None:
        pass

    @staticmethod
    def haversine_np(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        
        All args must be of equal length.    
        
        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6378137 * c
        return km

    @staticmethod
    def diagonal_length_min_box(min_box):
        x1, y1, x2, y2 = min_box
        pt1 = (x1, y1)
        pt2 = (x2, y1)
        pt4 = (x1, y2)

        dist12 = scipy.spatial.distance.euclidean(pt1, pt2)
        dist23 = scipy.spatial.distance.euclidean(pt1, pt4)
    
        return np.sqrt(dist12**2 + dist23**2)

    @staticmethod
    def intra_inter_region_transition(poi1, poi2):
        if poi1["GEOID"] == poi2["GEOID"]:
            return 1
        else:
            return 0.4

class Preprocess():
    def __init__(self, pois_filename, boroughs_filename, city) -> None:
        self.pois_filename = pois_filename
        self.boroughs_filename = boroughs_filename
        self.city = city

    def _read_poi_data(self):
        self.pois = pd.read_csv(self.pois_filename)
        self.pois["geometry"] = self.pois["geometry"].apply(wkt.loads)
        self.pois = gpd.GeoDataFrame(self.pois, geometry="geometry", crs="EPSG:4326")
        self.pois["geometry"] = self.pois["geometry"].apply(lambda x: x if x.geom_type == "Point" else x.centroid)
    
    def _read_boroughs_data(self):
        self.boroughs = pd.read_csv(self.boroughs_filename)
        self.boroughs["geometry"] = self.boroughs["geometry"].apply(wkt.loads)
        self.boroughs = gpd.GeoDataFrame(self.boroughs, geometry="geometry", crs="EPSG:4326")

        first_level = LabelEncoder()
        second_level = LabelEncoder()
        
        first_level.fit(self.pois["level_0"].values)
        second_level.fit(self.pois["level_1"].values)

        le_first_level_name_mapping = dict(zip(first_level.transform(first_level.classes_), first_level.classes_))
        le_second_level_name_mapping = dict(zip(second_level.transform(second_level.classes_), second_level.classes_))

        print(le_first_level_name_mapping)

        with open(f'../data/le_first_level_name_mapping_{self.city}.pkl', 'wb') as f:
            pkl.dump(le_first_level_name_mapping, f)
        
        with open(f'../data/le_second_level_name_mapping_{self.city}.pkl', 'wb') as f:
            pkl.dump(le_second_level_name_mapping, f)

        self.pois["category"] = first_level.fit_transform(self.pois["level_0"].values)
        self.pois["fclass"] = second_level.fit_transform(self.pois["level_1"].values)

        print(self.pois.shape)
        self.pois = self.pois[['id', 'category', 'fclass', 'geometry']].sjoin(self.boroughs, how='inner', predicate='intersects')
        print(self.pois.shape)
    def _read_embedding(self):
        from warnings import simplefilter
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        
        self.embedding_array = pd.get_dummies(self.pois['category'], dtype=int)
        self.embedding_array_test = []
        print(self.embedding_array.sum())

        G = nx.from_pandas_edgelist(self.edges)

        for i in range(len(self.embedding_array)):
            neighbors = []
            if G.has_node(i):
                neighbors = [n for n in G.neighbors(i)]

            if len(neighbors)==0:
                print(i)
                self.embedding_array_test.append([0]*7)
            else:
                # print(self.embedding_array.iloc[neighbors].mean(axis=0).values)
                self.embedding_array_test.append(self.embedding_array.iloc[neighbors].mean(axis=0).values)

        self.embedding_array_test = pd.DataFrame(self.embedding_array_test, columns=self.embedding_array.columns)
        
        # print(len(self.embedding_array), len(self.embedding_array_test))

    def _create_graph(self):
        # if os.path.exists('../data/edges.csv'):
        #     self.edges = pd.read_csv('../data/edges.csv')
        #     return
        
        D = Util.diagonal_length_min_box(self.pois.geometry.unary_union.envelope.bounds)

        coordinates = np.column_stack((self.pois.geometry.x, self.pois.geometry.y))
        cells, generators = voronoi_frames(coordinates, clip="extent")
        delaunay = weights.Rook.from_dataframe(cells)
        G = delaunay.to_networkx()
        positions = dict(zip(G.nodes, coordinates))
       
        for edges in G.edges:
            x, y = edges
            dist = Util.haversine_np(*positions[x], *positions[y])
            w1 = np.log((1+D**(3/2))/(1+dist**(3/2)))
            w2 = Util.intra_inter_region_transition(self.pois.iloc[x], self.pois.iloc[y])
            G[x][y]['weight'] = w1*w2
        
        self.edges = nx.to_pandas_edgelist(G)
        mi = self.edges['weight'].min()
        ma = self.edges['weight'].max()
        self.edges['weight'] = self.edges['weight'].apply(lambda x: (x-mi)/(ma-mi))

        self.edges.to_csv('../data/edges.csv', index=False)
    
    def get_data_torch(self):
        print("reading poi data")
        self._read_poi_data()
        
        print("reading boroughs data")
        self._read_boroughs_data()

        print("creating graph")
        self._create_graph()

        print("reading embedding")
        self._read_embedding()

        print("finishing preprocessing")
        
        data = {}
        data['edge_index'] = self.edges[["source", "target"]].T.values
        data['edge_weight'] = self.edges["weight"].values
        data['embedding_array'] = self.embedding_array.values
        data['embedding_array_test'] = self.embedding_array_test.values
        data['number_pois'] = len(self.embedding_array_test)
        data['y'] = self.pois['category'].values

        return data
    
if __name__ == "__main__":
    city = 'florida'

    pois_filename = f"../../data/pois_local_{city}.csv"
    boroughs_filename = f"../../data/cta_{city}.csv"
    # edges_filename = "../../poi-encoder/data/edges.csv"
    pre = Preprocess(pois_filename, boroughs_filename, city)
    data = pre.get_data_torch()
    # print(data)

    with open(f"../data/{city}_data.pkl", "wb") as f:
        pkl.dump(data, f)

    print("Data saved")

