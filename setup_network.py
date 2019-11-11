#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : setup_network.py 
# Creation  : 04 Mar 2016
# Time-stamp: <Fre 2016-07-22 15:18 juergen>
#
# Copyright (c) 2016 Jürgen Hackl <hackl@ibi.baug.ethz.ch>
#               http://www.ibi.ethz.ch
# $Id$ 
#
# Description : Set up test network
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 
# =============================================================================

# Load libraries
import timeit
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import mapping, Polygon, Point, LineString
import fiona

def distance(p0, p1):
    return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


class TestNetwork(object):
    """Random River Netork"""

    def __init__(self):
        self.G = None

        # set up shp information
        self.source_driver = 'ESRI Shapefile'
        self.source_crs = {'no_defs': True, 'ellps': 'WGS84', 'datum': 'WGS84', 'proj': 'longlat'}
        # self.source_crs = {'lat_0': 46.95240555555556, 'proj': 'somerc', 'lon_0': 7.439583333333333, 'x_0': 600000, 'units': 'm', 'ellps': 'bessel', 'no_defs': True, 'y_0': 200000, 'k_0': 1}
        pass

        
    def create_roads(self):
        self.roads = nx.DiGraph()
        self.coord_dict = {'a':(3000,0),
                           'b':(6000,4000),
                           'c':(9000,0),
                           'd':(6000,10000),
                           'e':(12000,18000),
                           'f':(6000,18000),
                           'g':(0,18000)}

        roads_list = [['a','c',1,'2_Klass','Road',900,None,80,0,0],
                      ['a','b',2,'2_Klass','Road',900,None,80,0,0],
                      ['c','b',3,'2_Klass','Road',900,None,80,0,0],
                      ['b','d',4,'A_Klass','Bridge',2000,None,120,0,2],
                      ['d','f',5,'1_Klass','Road',1200,None,100,0,0],
                      ['d','e',6,'3_Klass','Road',600,None,50,0,2],
                      ['d','g',7,'3_Klass','Road',600,None,50,0,1],
                      ['g','f',8,'2_Klass','Road',900,None,80,0,0],
                      ['e','f',9,'2_Klass','Road',900,None,80,0,0],
                      ['a','g',10,'3_Klass','Road',600,None,50,1,2]]

        # roads_list = [['a','c',1,'2_Class',900,None,80,0,0],
        #               ['a','b',2,'2_Class',900,None,80,0,0],
        #               ['c','b',3,'2_Class',900,None,80,0,0],
        #               ['b','d',4,'A_Class',2000,None,120,0,0],
        #               ['d','f',5,'1_Class',1200,None,100,0,0],
        #               ['d','e',6,'3_Class',600,None,50,0,0],
        #               ['d','g',7,'3_Class',600,None,50,0,0],
        #               ['g','f',8,'2_Class',900,None,80,0,0],
        #               ['e','f',9,'2_Class',900,None,80,0,0]]


        
        for road in roads_list:
            if road[6] == None:
                road_length = distance(self.coord_dict[road[0]],self.coord_dict[road[1]])
            else:
                road_length = road[6]
            self.roads.add_edge(self.coord_dict[road[0]],self.coord_dict[road[1]],name=str(road[2])+':'+road[0]+'-'+road[1],type=road[3],object=road[4],capacity=road[5], length=road_length,speedlimit=road[7],oneway=road[8],damage=road[9])
            if road[8] == 0:
                self.roads.add_edge(self.coord_dict[road[1]],self.coord_dict[road[0]],name=str(road[2])+':'+road[1]+'-'+road[0],type=road[3],object=road[4],capacity=road[5], length=road_length,speedlimit=road[7],oneway=road[8],damage=road[9])
        pass

    def create_centroids(self):
        self.centroids_dict = {'0':(3000,1000),
                               '1':(9000,1000),
                               '2':(0,19000),
                               '3':(12000,19000)}
        self.connections_list = [('0','a'),('1','c'),('2','g'),('3','e')]

    def save_roads(self,filename):
        """Save network to a shp file"""

        G = self.roads
        # Write a new Shapefile for the edges
        source_schema = {
            'geometry': 'LineString',
            'properties': {'name': 'str', 'type': 'str', 'object': 'str', 'capacity': 'int', 'speedlimit': 'int', 'length': 'float', 'oneway':'int', 'damage':'int'},
        }
        with fiona.open(filename + '.shp', 'w', driver=self.source_driver, crs=self.source_crs, schema=source_schema) as output:
            ## If there are multiple geometries, put the "for" loop here
            for edge in G.edges():
                u = edge[0]
                v = edge[1]
                line = LineString([u,v])
                output.write({
                    'geometry': mapping(line),
                    'properties': {'name': G[u][v]['name'], 'type': G[u][v]['type'], 'object': G[u][v]['object'],
                                   'capacity': G[u][v]['capacity'], 'speedlimit': G[u][v]['speedlimit'],
                                   'length': G[u][v]['length'], 'oneway': G[u][v]['oneway'],'damage': G[u][v]['damage']},
                })
        pass


    def save_centroids(self,filename):
        """Save points to a shp file"""

        # Write a new Shapefile for the nodes
        source_schema = {
            'geometry': 'Point',
            'properties': {'name': 'int'},
        }
        with fiona.open(filename + '.shp', 'w', driver=self.source_driver, crs=self.source_crs, schema=source_schema) as output:
            ## If there are multiple geometries, put the "for" loop here
            for idx,coords in self.centroids_dict.items():
                p = Point([coords])
                output.write({
                    'geometry': mapping(p),
                    'properties': {'name': int(idx)},
                })

        pass

    def save_connections(self,filename):
        """Save connection lines to a shp file"""

        # Write a new Shapefile for the edges
        source_schema = {
            'geometry': 'LineString',
            'properties': {'id': 'int'},
        }
        with fiona.open(filename + '.shp', 'w', driver=self.source_driver, crs=self.source_crs, schema=source_schema) as output:
            ## If there are multiple geometries, put the "for" loop here
            for line in self.connections_list:
                u = self.centroids_dict[line[0]]
                v = self.coord_dict[line[1]]
                idx = int(line[0])
                line = LineString([u,v])
                output.write({
                    'geometry': mapping(line),
                    'properties': {'id': idx},
                })
        pass


    def create_od_matrix(self):
        self.od_matrix = np.asarray([ [100,400,400,500], [350,100,300,400], [200,200,200,500], [200,200,700,300] ])*2

    def save_od_matrix(self,filename):
        np.savetxt(filename+'.csv', self.od_matrix, delimiter=",")

    def test(self):
        """test function"""
        self.create_roads()
        self.save_roads('./temp/roads')
        self.create_centroids()
        self.save_centroids('./temp/centroids')
        self.save_connections('./temp/connections')
        self.create_od_matrix()
        self.save_od_matrix('./temp/od')

# print('start')
# network = TestNetwork()
# network.test()
# print('end')
        

# =============================================================================
# eof
#
# Local Variables: 
# mode: python
# End: 

 
