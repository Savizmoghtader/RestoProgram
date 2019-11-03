#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : trafficmodel.py
# Creation  : 14 Jun 2017
# Time-stamp: <Wed 2019-10-30 14:49 juergen>
#
# Copyright (c) 2017 Jürgen Hackl <hackl@ibi.baug.ethz.ch>
#               http://www.ibi.ethz.ch
# $Id$
#
# Description : A simple traffic model based on Frank-Wolfe
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

import os
import subprocess
import networkx as nx
import numpy as np
import itertools
import tempfile
from shutil import rmtree

WINDOWS = True

class TrafficSimulation(object):
    def __init__(self, graph=None, od_graph=None):

        self.graph = graph
        self.od_graph = od_graph
        self.print_flag = True
        self.unit_factor = 1000
        pass

    def get_edge_attribute(self, edge, attribute):
        return self.graph[edge[0]][edge[1]][attribute]

    def set_edge_attribute(self, edge, attribute, value):
        self.graph[edge[0]][edge[1]][attribute] = value
        pass

    def calculate_initial_traveltime(self):
        for edge in self.graph.edges():
            initial_traveltime = self.get_edge_attribute(
                edge, 'length') / self.unit_factor / self.get_edge_attribute(edge, 'speedlimit')
            self.set_edge_attribute(edge, 't_0', initial_traveltime)
        pass

    def set_initial_traveltimes(self):
        for edge in self.graph.edges():
            self.set_edge_attribute(
                edge, 't_k', self.get_edge_attribute(edge, 't_0'))
            self.set_edge_attribute(edge, 't_h', 0)
        pass

    def set_initial_flow(self):
        for edge in self.graph.edges():
            self.set_edge_attribute(edge, 'flow', 0)
        pass

    def set_initial_help_flow(self):
        for edge in self.graph.edges():
            self.set_edge_attribute(edge, 'help_flow', 0)
        pass

    def set_od_matix(self, od_matrix):
        for edge in self.od_graph.edges():
            s = edge[0]
            t = edge[1]
            self.od_graph[s][t]['demand'] = od_matrix[s, t]
        pass


class TrafficModel(object):
    def __init__(self, graph, od_graph, od_matrix=None):
        self.graph = graph.copy()
        self.od_graph = od_graph.copy()
        self.od_matrix = np.copy(od_matrix)
        self.lost_trips = {}

        self.print_flag = True
        temp = True
        self.temp_folder = './traffic_fw/temp/'
        # self.temp_folder = './temp/'
        if temp:
            self.temp_dir = tempfile.mkdtemp(dir=self.temp_folder)
            self.temp_name = os.path.split(self.temp_dir)[-1]
        else:
            self.temp_dir = self.temp_folder+'XXX/'
            self.temp_name = 'XXX'

        self.alpha = 0.15
        self.beta = 4
        self.toll = 0
        self.unit_factor = 1000
        self.threshold = 1e-4
        self.large = 1e+14
        # number of iterations for the corrective factors
        self.n_iter_tm = 1000  # 1000
        pass

    def get_edge_attribute(self, edge, attribute):
        return self.graph[edge[0]][edge[1]][attribute]

    def set_edge_attribute(self, edge, attribute, value):
        self.graph[edge[0]][edge[1]][attribute] = value
        pass

    def set_od_matix(self, od_matrix):
        for edge in self.od_graph.edges():
            s = edge[0]
            t = edge[1]
            self.od_graph[s][t]['demand'] = od_matrix[s, t]
        pass

    def check_network_connections(self):
        graph = self.graph.copy()

        for edge in list(graph.edges()):
            if self.get_edge_attribute(edge, 'capacity') == 1:
                graph.remove_edge(edge[0], edge[1])

        for edge in list(self.od_graph.edges()):
            s = edge[0]  # source
            t = edge[1]  # target
            if not nx.has_path(graph, s, t):
                self.lost_trips[(s, t)] = self.od_graph[s][t]['demand']
                self.od_graph.remove_edge(s, t)

        for node in list(self.od_graph.nodes()):
            if self.od_graph.degree(node) <= 2:
                mapping = {node: self.od_graph.node[node]['coordinates']}
                self.od_matrix = np.delete(
                    np.delete(self.od_matrix, node, 0), node, 1)
                self.od_graph.remove_node(node)
                self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)
        pass

    def process_network(self):
        nodes = list(self.od_graph.nodes())
        nodes.sort()
        mapping = {node: i+1 for i, node in enumerate(nodes)}
        self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)
        self.od_graph = nx.relabel_nodes(self.od_graph, mapping, copy=False)

        k = nodes[-1] + 1
        mapping = {node: i+k for i,
                   node in enumerate(list(set(self.graph.nodes())-set(self.od_graph.nodes())))}
        self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)

        number_of_zones = self.od_graph.number_of_nodes()
        number_of_nodes = self.graph.number_of_nodes()
        first_thru_node = 1
        number_of_links = self.graph.number_of_edges()

        net_rows = ['<NUMBER OF ZONES> ' + str(number_of_zones),
                    '<NUMBER OF NODES> ' + str(number_of_nodes),
                    '<FIRST THRU NODE> ' + str(first_thru_node),
                    '<NUMBER OF LINKS> ' + str(number_of_links),
                    '<END OF METADATA> \n',
                    '~\t Init node \t Term node \t Capacity \t Length \t Free Flow Time \t B \t Power \t Speed limit \t Toll \t Link Type \t'
                    ]

        for e in self.graph.edges(data=True):
            attr = e[2]
            line = [e[0],                                                # Init node
                    # Term node
                    e[1],
                    # Capacity (veh/h)
                    attr['capacity'],
                    # Length (km)
                    attr['length']/self.unit_factor,
                    attr['length']/self.unit_factor / \
                    attr['speedlimit'],  # Free Flow Time (h)
                    self.alpha,                                          # B
                    self.beta,                                           # Power
                    # Speed limit (km/h)
                    attr['speedlimit'],
                    self.toll,                                           # Toll
                    # Link Type
                    attr['oneway']
                    ]
            net_rows.append('\t'+'\t'.join(map(str, line))+'\t;')

        with open(self.temp_dir+'/'+self.temp_name+'_net.tntp', 'w') as net_file:
            net_file.write('\n'.join(net_rows))

        total_od_flow = np.sum(self.od_matrix)

        width_1 = len(str(number_of_zones))+2
        width_2 = len(str(int(np.max(self.od_matrix))))+5

        trips_rows = ['<NUMBER OF ZONES> '+str(number_of_zones),
                      '<TOTAL OD FLOW> '+str(total_od_flow),
                      '<END OF METADATA>'
                      ]
        for i, o in enumerate(self.od_matrix):
            trips_rows.append('\nOrigin '+str(i+1))
            col = 0
            line = []
            for j, d in enumerate(o):
                line.append("{0:>{2}} :{1:>{3}.{digits}f};".format(
                    j+1, d, width_1, width_2, digits=2))
            n = 5
            lines = [line[i:i + n] for i in range(0, len(line), n)]
            [trips_rows.append(''.join(l)) for l in lines]

        with open(self.temp_dir+'/'+self.temp_name+'_trips.tntp', 'w') as trips_file:
            trips_file.write('\n'.join(trips_rows))
        pass

    def ta_frank_wolf(self):
        args = [os.path.abspath(self.temp_folder+self.temp_name), self.temp_name,
                str(self.threshold), str(self.n_iter_tm)]

        if WINDOWS:
            comand_julia = 'C:\\Users\\smoghtade\\AppData\\Local\\Julia-1.2.0\\bin\\julia '
        else:
            comand_julia = 'julia '
        output = subprocess.check_output(
            #'julia ./traffic_fw/ta.jl '+' '.join(args), shell=True).decode("utf-8").strip('\n')
        #'C:\\Users\\smoghtade\\AppData\\Local\\Julia-1.2.0\\bin\\julia ./traffic_fw/ta.jl ' + ' '.join(args), shell = True).decode("utf-8").strip('\n')
            comand_julia + './traffic_fw/ta.jl ' + ' '.join(args), shell = True).decode("utf-8").strip('\n')
        # output = subprocess.check_output('julia ta.jl '+' '.join(args), shell=True).decode("utf-8").strip('\n')
        values = [float(x) for x in output.split()]
        self.flow = values[0]
        self.traveltime = values[1]
        self.car_hours = values[2]
        self.car_distances = values[3]
        pass

    def get_traveltime(self):
        return [self.traveltime]

    def get_flow(self):
        return [self.flow]

    def get_car_hours(self):
        return [self.car_hours]

    def get_car_distances(self):
        return [self.car_distances]

    def get_lost_trips(self):
        return self.lost_trips

    def run(self):

        # print(nx.info(self.od_graph))
        # assign od matrix to od graph (if matrix is given)
        if self.od_matrix is not None:
            self.set_od_matix(self.od_matrix)

        # check network if every source and target can be reached
        self.check_network_connections()

        # # create input file for the simulation
        self.process_network()

        # run frank-wolf algorithm
        self.ta_frank_wolf()

        # clean folders
        #os.system('rm -rf ' + self.temp_dir)
        rmtree(self.temp_dir)
        pass

# import timeit

# def test():
#     print('***** start *****')
#     start = timeit.default_timer()

#     ### Initialization of the graphs ###

#     # road graph
#     road_graph = read_shp('./data/road_simple.shp')
#     #road_graph = read_shp('./temp/roads.shp')
#     #road_graph = read_shp('./temp/roads_damaged.shp')

#     # od graph (without external od matrix)
#     od_graph = create_od_graph('./data/centroids.shp')
#     #od_graph = create_od_graph('./temp/centroids.shp')

#     # load connections from od nodes to road network
#     con_edges = read_shp('./data/connections.shp')
#     #con_edges = read_shp('./temp/connections.shp')

#     # create network for traffic model
#     graph = create_network_graph(road_graph,od_graph,con_edges)

#     ## Traffic Model ##

#     # load given od matix
#     od_matrix = np.genfromtxt('./data/od_0.csv', delimiter=',')
#     #od_matrix = np.genfromtxt('./temp/od.csv', delimiter=',')

#     ### REMARK ###
#     # Final flow results has to be multiplied wit 0.46
#     # for a hourly analysis and with 10.9 for a daily analysis !!!

#     start_run = timeit.default_timer()
#     # set up traffic model
#     traffic = TrafficModel(graph,od_graph,od_matrix)

#     # run traffic simulation
#     traffic.run()

#     # get the new graph of roads
#     #graph_result = traffic.get_graph()

#     print(sum(traffic.get_traveltime()))
#     print(sum(traffic.get_flow()))
#     print(sum(traffic.get_car_hours()))
#     print(sum(traffic.get_car_distances()))

#     end_run = timeit.default_timer()

#     print('run time: ' + str(end_run - start_run))
#     stop = timeit.default_timer()
#     print('total time: ' + str(stop - start))
#     print('****** end ******')

# test()

# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# End:
