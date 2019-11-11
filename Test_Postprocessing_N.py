#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : main.py
# Creation  : 11 Mar 2016
# Time-stamp: <Die 2019-06-11 17:16 juergen>
#
# Copyright (c) 2016 Jürgen Hackl <hackl@ibi.baug.ethz.ch>
#               http://www.ibi.ethz.ch
# $Id$
#
# Description : main file
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
import networkx as nx
import numpy as np
import random
import pickle
import ast
import itertools

from operator import itemgetter

from setup_network import TestNetwork

from damagemodel import DamageModel
from restorationmodel import RestorationModel

from traffic_fw.trafficmodel import TrafficModel
from traffic_fw.initialize import *
#from traffic_fw.graph2shp import write_shp
#from graph2shp import save_shp
from simanneal import Annealer

from joblib import Parallel, delayed

import timeit
from ToolBox import *


class PpEngine(object):
    """ PostprocessingSimulation engine
    """

    def __init__(self, filepath='./'):
        self.filepath = filepath
        self.test = True#False#True
        self.a = 25.5
        self.b = 10
        self.c = 1240

        self.mu = np.array([0.94, 0.06])
        self.xi = np.array([23.02, 130.96])

        self.F_w = np.array([6.7, 33])/100
        self.nu = 1.88
        self.rho = np.array([14.39, 32.54])/100
        self.upsilon = 83.27 * 8
        self.day_factor = 9
        self.capacity_losses = {'Bridge': {0: 0, 1: .5, 2: 1, 3: 1}, 'Road': {
            0: 0, 1: .7, 2: 1, 3: 1}, 'Tunnel': {0: 0}}

        self.gamma = 1
        # type 1: normal
        # type 2: emergency
        # type 3: partial
        # TODO: Load data from csv file
        self.restoration_names = {
            0: 'high priority', 1: 'normal', 2: 'low priority'}
        self.restoration_types = [0, 1, 2]
        # self.restoration_types = [2,2,2]

    def create_network(self):
        self.network = TestNetwork()
        self.network.test()
        pass

    def initialize_network(self):
        if self.test:
            self.road_graph = read_shp('./temp/roads.shp')
            self.od_graph = create_od_graph('./temp/centroids.shp')
            self.con_edges = read_shp('./temp/connections.shp')
            self.od_matrix = np.genfromtxt('./temp/od.csv', delimiter=',')
        else:
            # self.road_graph = read_shp('./data/roads_dir.shp')
            self.road_graph = read_shp('./data/roads_clean.shp')
            self.od_graph = create_od_graph('./data/centroids.shp')
            self.con_edges = read_shp('./data/connections.shp')
            self.od_matrix = np.genfromtxt('./data/od.csv', delimiter=',')

        self.graph = create_network_graph(
            self.road_graph, self.od_graph, self.con_edges)
        pass

    def initialize_damage(self):
        self.damage = DamageModel(self.graph, self.capacity_losses)
        self.damage.run()
        self.graph_damaged = self.damage.get_graph()
        self.damage_dict = self.damage.get_damage_dict(directed=False)
        #self.damage_dict = self.damage.get_damage_dict(directed=True)
        pass

    def run_restoration_model(self, sequence=None):
        self.restoration = RestorationModel(self.graph_damaged, self.filepath)
        self.restoration.run(sequence)
        self.restoration_graphs = self.restoration.get_restoration_graphs()
        self.restoration_times = self.restoration.get_restoration_times()
        self.restoration_costs = self.restoration.get_restoration_costs()
        pass

    def run_traffic_model(self, graph, od_graph):
        # set up traffic model
        #self.traffic = TrafficModel(graph.to_undirected(),od_graph.to_undirected(),self.od_matrix)
        self.traffic = TrafficModel(graph, od_graph, self.od_matrix)
        # run traffic simulation
        self.traffic.run()
        # self.traffic.print_results()
        t_k = sum(self.traffic.get_traveltime())
        flow = sum(self.traffic.get_flow())
        hours = sum(self.traffic.get_car_hours())
        distances = sum(self.traffic.get_car_distances())
        lost_trips = sum(self.traffic.get_lost_trips().values())
        # graph_result = self.traffic.get_graph()
        return t_k, flow, hours, distances, lost_trips

    # def save_results(self,path_shp):
    #     save_shp(self.graph_result,path_shp)
    #     pass

    def initialize_state(self):
        init_edges = list(self.damage_dict.keys())
        random.shuffle(init_edges)

        init_state = []
        # ran_seq = [0,1,2,2]
        # for i,edge in enumerate(init_edges):
        #     init_state.append((edge,ran_seq[i]))

        for edge in init_edges:
            init_state.append((edge, random.choice(self.restoration_types)))
        #    #     init_state.append((edge,0))
        return init_state

    # def parallel_model(self,graph):
    #     g = graph.copy()
    #     od_graph = self.od_graph.copy()
    #     damaged = self.run_traffic_model(g,od_graph)
    #     return get_delta(default,damaged)

    def test_run_restoration(self):
        print('--- initialization ---')
        self.create_network()
        self.initialize_network()
        self.initialize_damage()
        #init_state = self.initialize_state()
        init_state = [(((757191.29, 190611.49), (757051.9, 190757.8)), 2), (((759564.7, 193863.6), (759486.2, 193874.4)), 0), (((750173.5, 187788.7), (750254.0, 187818.8)), 1), (((760072.86, 195139.91), (760316.99, 195480.22)), 2), (((750029.5, 187527.68), (750111.3, 187316.8)), 2), (((758869.98, 192919.98), (759283.67, 193198.51)), 2), (((759486.67, 194199.2), (759684.8, 194851.49)), 2), (((750418.28, 187694.49), (750372.0, 187763.7)), 2), (((750758.67, 185809.8), (750687.71, 185933.99)), 2), (((750205.1, 187513.1), (750274.52, 187591.83)), 2), (((759199.86, 192428.21), (759496.12, 193153.65)), 2), (((750892.8, 188259.0), (750846.7, 188266.5)), 2), (((761299.9, 197672.1), (760157.04, 195601.1)), 2), (((757295.4, 191208.2), (756999.6, 191494.5)), 2), ((
            (750274.52, 187591.83), (750970.7, 188290.6)), 2), (((750846.7, 188266.5), (750720.3, 188285.3)), 2), (((759684.8, 194851.49), (759921.1, 195903.9)), 2), (((757048.2, 190761.8), (756692.3, 191010.4)), 2), (((759276.93, 193205.89), (758864.56, 192928.38)), 2), (((757051.9, 190757.8), (757384.9, 191117.0)), 2), (((756566.83, 191085.62), (757125.6, 191859.3)), 2), (((760146.99, 195565.56), (760061.78, 195228.28)), 2), (((750205.1, 187513.1), (750032.6, 187537.7)), 2), (((760653.9, 196792.7), (761122.2, 197363.1)), 2), (((756566.7, 190985.3), (756566.83, 191085.62)), 2), (((756263.0, 190580.5), (756566.7, 190985.3)), 2), (((750183.5, 187097.9), (750024.54, 187597.17)), 2), (((757295.4, 191208.2), (757357.01, 191276.99)), 2), (((760395.1, 196965.5), (760902.2, 197368.6)), 2)]
        print('--- restoration ---')
        self.run_restoration_model(init_state)

        print(self.restoration_times)
        pass

    def load_state(self, filename):
        with open(filename, 'r') as f:
            state = ast.literal_eval(f.read())
        return state

    def load_damage(self, filename):
        with open(filename, 'r') as f:
            damaged = ast.literal_eval(f.read())
        return damaged

    def load_costs(self, filename):
        with open(filename, 'r') as f:
            costs = ast.literal_eval(f.read())
        return costs

    def calculate_damage(self):
        print('--- initialization ---')
        self.initialize_network()
        self.initialize_damage()

        # init_state = self.initialize_state()
        init_state = self.load_state(self.filepath + 'state.txt')

        print('--- no and initial damage ---')

        no_damage = self.run_traffic_model(self.graph, self.od_graph)
        initial_damage = self.run_traffic_model(
            self.graph_damaged.copy(), self.od_graph.copy())

        self.damaged = []
        self.damaged.append(get_delta(no_damage, initial_damage))

        print('--- restoration ---')
        self.run_restoration_model(init_state)

        print('--- simulated annealing ---')
        sim_results = Parallel(n_jobs=6)(delayed(parallel_model)(
            graph, self.od_graph, self.od_matrix) for graph in self.restoration_graphs[:-1])

        for damaged in sim_results:
            self.damaged.append(get_delta(no_damage, damaged))

        with open(self.filepath + 'damaged.txt', 'w') as f:
            f.write(str(self.damaged))

        pass

    def run(self, scenario):
        print('--- initialization ---')
        self.initialize_network()
        self.initialize_damage()
        init_state = self.load_state(self.filepath + 'state.txt')

        print('--- calculate damage ---')
        if not os.path.isfile(self.filepath + 'damaged.txt'):
            self.calculate_damage()
        else:
            self.damaged = self.load_damage(self.filepath + 'damaged.txt')

        # t_k, flow, hours, distances, lost_trips
        names_dict = {}
        for object in init_state:
            names_dict[object[0]] = get_edge_attribute(
                self.graph, object[0], 'name')+'-' + get_edge_attribute(self.graph, object[0], 'object')[0]

        self.restoration_results = RestorationModel(
            self.graph_damaged, self.filepath)
        # (object,schedule time,needed time,#resources,intervention type, assignd resource)
        sequence_formated = self.restoration_results.format(init_state)

        sort_names = []
        for s in sequence_formated:
            for t in s:
                sort_names.append(
                    [t[1]-t[2], t[-1], get_edge_attribute(self.graph, t[0], 'name')[2:]])

        # remove dopples
        seen = set()
        sort_names = [x for x in sort_names if x[2]
                      not in seen and not seen.add(x[2])]
        sort_names = sorted(sort_names, key=itemgetter(1))
        sort_names = sorted(sort_names, key=itemgetter(0))

        name_dict = {}
        for name, key in enumerate(sort_names):
            name_dict[key[2]] = str(name+1)

#        print(name_dict)
        name_dict = {'1095': '6', '460': '10', '1703': '14', '1237': '20', '562': '11', '2069': '12', '1798': '24', '471': '28', '1371': '29', '1692': '15', '2043': '26', '1802': '8', '1907': '18', '1233': '23',
                     '1913': '13', '2042': '1', '2052': '7', '2131': '17', '554': '2', '1276': '21', '1202': '22', '1706': '9', '1803': '5', '1905': '19', '1279': '16', '1814': '25', '332': '3', '1498': '27', '461': '4'}
        i = 0
        lines = []

        name_list = []
        for s in sequence_formated:
            for t in s:
                name_list.append(get_edge_attribute(
                    self.graph, t[0], 'name')[2:])

        new_name_list = []
        for name in name_list:
            if name not in new_name_list:
                new_name_list.append(name)

        name_list = list(set(name_list))

        program_list = []
        for s in sequence_formated:
            for t in s:
                crew = t[5]
                task_end = t[1]
                task_duration = t[2]
                task_start = task_end - task_duration
                object_type = get_edge_attribute(self.graph, t[0], 'object')
                damage = get_edge_attribute(self.graph, t[0], 'damage')

                condition_state = {1: 'cs1', 2: 'cs2'}
                intervention_type = {
                    10: 'cs1i0', 11: 'cs1i1', 12: 'cs1i2', 20: 'cs2i0', 21: 'cs2i1', 22: 'cs2i2'}
                resources = {1: 'res1', 2: 'res2'}
                x_scale = .4012/2
                y_scale = .45
                start_point = (task_start * x_scale, crew * y_scale)
                end_point = (task_end * x_scale, (crew+1) * y_scale)
                name = get_edge_attribute(self.graph, t[0], 'name')[2:]
                lines.append('\draw[normal,'+resources[t[3]]+','+condition_state[damage]+','+intervention_type[int(str(damage)+str(t[4]))]+'] '+str(
                    start_point)+' rectangle '+str(end_point) + ' node[mynode,pos=.5] {'+name_dict[name]+' \\\ '+object_type[0]+'};')
                #lines.append('\draw[normal,'+resources[t[3]]+','+condition_state[damage]+','+intervention_type[int(str(damage)+str(t[4]))]+'] '+str(start_point)+' rectangle '+str(end_point)+ ' node[mynode,pos=.5] {'+name+' \\\ '+object_type[0]+'};')

                condition_state = {1: '1 minor', 2: '2 major'}
                intervention_type = {10: 'level 1', 11: 'level 2',
                                     12: 'level 3', 20: 'level 1', 21: 'level 2', 22: 'level 3'}
                intervention_nr = {10: 'i', 11: 'ii',
                                   12: 'iii', 20: 'i', 21: 'ii', 22: 'iii'}
                crew_type = {0: 'A', 1: 'B', 2: 'C'}
                sp = ' & '
                nl = '\\'

                program_list.append([int(name_dict[name]), name, object_type, condition_state[damage], intervention_type[int(
                    str(damage)+str(t[4]))], str(task_start/2), str(task_end/2), str(task_duration/2), crew_type[crew]])
                # print(name_dict[name]+sp+name+sp+object_type+sp+condition_state[damage]+sp+intervention_nr[int(str(damage)+str(t[4]))]+sp+intervention_type[int(str(damage)+str(t[4]))]+sp+str(task_start/2)+sp+str(task_end/2)+sp+str(task_duration/2)+sp+crew_type[crew]+sp)
                i += 1

        costs_list = self.load_costs(self.filepath + 'costs.txt')
        for row in costs_list:
            del row[0]

        for i, l in enumerate(program_list):
            a = int(round(costs_list[i][0], 0))
            b = int(round(costs_list[i][1], 0))
            c = int(round(costs_list[i][2], 0))
            d = a+b+c
            l.extend([a, b, c, d])

        seen = set()
        program = [x for x in program_list if x[0]
                   not in seen and not seen.add(x[0])]

        sum_fixed = sum(row[-4] for row in program)
        sum_variable = sum(row[-3] for row in program)
        sum_resources = sum(row[-2] for row in program)
        sum_total = sum(row[-1] for row in program)

        program.sort(key=lambda x: x[0])

        program.append(['', '', '', '', '', '', '', '', '',
                        sum_fixed, sum_variable, sum_resources, sum_total])
        with open(self.filepath + 'tex/program_0'+scenario+'.tex', mode='wt', encoding='utf-8') as f:
            for p in program:
                for i in range(9, 13):
                    p[i] = '\\numprint{'+str(p[i])+'}'
                f.write(' & '.join(str(e) for e in p)+' \\\ \n')

        with open('./results/tex/resources_0'+scenario+'.tex', mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(lines))

        with open(self.filepath + 'tex/resources_01.tex', mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(lines))

        #print('t_k, flow, hours, distances, lost_trips')
        # for damage in damaged:
        #     print(damage)
        fix_costs = [0]
        variable_costs = [0]
        resources_costs = [0]
        all_costs = [0]
        time = [0]

        self.run_restoration_model(init_state)

#        print(self.restoration_costs)

        # print(self.damaged)
        result_matrix = np.zeros((9, len(self.damaged)+1))
        result_matrix_t = np.zeros((3, len(self.damaged)+1))
        x = np.zeros(len(self.damaged)+1)

        self.consequences = []
        for idx, damage in enumerate(self.damaged):
            if idx == 0:
                delta_t = self.restoration_times[idx]
            else:
                delta_t = (
                    self.restoration_times[idx]-self.restoration_times[idx-1])

            result_matrix[0][idx+1] = self.restoration_costs[idx][0]
            result_matrix[1][idx+1] = self.restoration_costs[idx][1]
            result_matrix[2][idx+1] = self.restoration_costs[idx][2]

            result_matrix[4][idx+1] = (damage[2] *
                                       np.sum(self.mu*self.xi))*delta_t * self.day_factor
            result_matrix[5][idx+1] = (damage[3] * np.sum(self.mu *
                                                          (self.nu * self.F_w + self.rho)))*delta_t
            result_matrix[6][idx+1] = (damage[4] * self.upsilon)*delta_t

            result_matrix_t[0][idx+1] = damage[2]
            result_matrix_t[1][idx+1] = damage[3]
            result_matrix_t[2][idx+1] = damage[4]

#            print(damage[2], delta_t * (damage[2] * np.sum(self.mu*self.xi)))
#            print(damage[3], delta_t * (damage[3] * np.sum(self.mu * (self.nu * self.F_w + self.rho))))
#            print(damage[4], delta_t * (damage[4] * self.upsilon))
#            print(damage[2])
            x[idx+1] = self.restoration_times[idx]

            self.consequences.append(self.gamma * delta_t * (self.day_factor * damage[2] * np.sum(self.mu*self.xi) + damage[3] * np.sum(
                self.mu * (self.nu * self.F_w + self.rho)) + damage[4] * self.upsilon) + sum(self.restoration_costs[idx]))

        print('consequences: ', sum(self.consequences))

        result_matrix_t[0:3, 0] = result_matrix_t[0:3, 1]
        result_matrix[3] = np.sum(result_matrix[0:3, :], axis=0)
        #result_matrix[4:7,0] = result_matrix[4:7,1]
        result_matrix[7] = np.sum(result_matrix[4:7, :], axis=0)
        result_matrix = np.cumsum(result_matrix, axis=1)
#        result_matrix_id = np.cumsum(result_matrix_id,axis=1)
        # result_rev = np.fliplr(result_matrix_id)
        # result_rev = np.cumsum(result_rev,axis=1)
        # result_matrix_id = np.fliplr(result_rev)

        print('direct:', result_matrix[3, -1])

        print(result_matrix[3, -1]+result_matrix[7, -1])
        # print(result_matrix[7,-1])
#        print(result_matrix)
        result_matrix[8] = result_matrix[3]+result_matrix[7]

        lines = []
        for i in range(4):
            lines.append(
                '\\addplot+[const plot, no marks, thick] coordinates {')
            for j in range(result_matrix.shape[1]):
                lines.append('('+str(x[j])+','+str(result_matrix[i, j])+')')
            lines.append('};')

        for i in range(4, 8):
            xvals = np.arange(0, x[-1]+0.5, 0.5)
            y = result_matrix[i]
            yinterp = np.interp(xvals, x, y)

            lines.append(
                '\\addplot+[const plot, no marks, thick] coordinates {')
            for j in range(xvals.shape[0]):
                lines.append('('+str(xvals[j])+','+str(yinterp[j])+')')
            lines.append('};')

        with open('./results/tex/costs_0'+scenario+'.tex', mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(lines))

        with open(self.filepath + 'tex/costs_01.tex', mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(lines))

        xvals = np.arange(0, x[-1]+0.5, 0.5)
        dc_vec = np.zeros(xvals.shape[0])
        for i in range(xvals.shape[0]):
            try:
                dc_vec[i] = result_matrix[3, list(x).index(xvals[i])]
            except ValueError:
                pass

        for i in range(dc_vec.shape[0]-1):
            if dc_vec[i+1] == 0.0 and dc_vec[i] > 0.0:
                dc_vec[i+1] = dc_vec[i]

        y = result_matrix[7]
        ic_vec = np.interp(xvals, x, y)

        vecs = [dc_vec, ic_vec, ic_vec+dc_vec]

        lines = []
        for vec in vecs:
            if scenario == '1':
                lines.append(
                    '\\addplot+[const plot, mymark={text}{text mark=1,text mark as node, text mark style={font=\\tiny,circle,inner sep=0pt,fill=myblue!10!white,},}, thick] coordinates {')
            elif scenario == '2':
                lines.append(
                    '\\addplot+[const plot, mymark={text}{text mark=2,text mark as node, text mark style={font=\\tiny,circle,inner sep=0pt,fill=damage2!10!white,},}, thick] coordinates {')
            else:
                lines.append(
                    '\\addplot+[const plot, mymark={text}{text mark=3,text mark as node, text mark style={font=\\tiny,circle,inner sep=0pt,fill=mygreen!10!white,},}, thick] coordinates {')
#            lines.append('\\addplot+[const plot, no marks, thick] coordinates {')
            for j in range(vec.shape[0]):
                lines.append('('+str(xvals[j])+','+str(vec[j])+')')
            lines.append('};')

        with open('./results/tex/sum_0'+scenario+'.tex', mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(lines))

        with open(self.filepath + 'tex/sum_01.tex', mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(lines))

        #x = np.append(x,x[-1])
        mat1 = np.append(result_matrix_t[0], result_matrix_t[0][-1])
        # for i in range(len(mat1)-1):
        #     print(mat1[i+1],x[i])

        mat2 = np.append(result_matrix_t[2], result_matrix_t[2][-1])
        # for i in range(len(mat2)-1):
        #     print(mat2[i+1],x[i])

        mat = np.zeros((2, len(mat1)-1))
        for i in range(len(mat1)-1):
            mat[0][i] = mat1[i+1]
            mat[1][i] = mat2[i+1]

        lines = []
        for i in [0, 1]:
            lines.append(
                '\\addplot+[const plot, no marks, thick] coordinates {')
            for j in range(mat.shape[1]):
                lines.append('('+str(x[j])+','+str(mat[i, j])+')')
            lines.append('};')

        with open('./results/tex/los_0'+scenario+'.tex', mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(lines))

        with open(self.filepath + 'tex/los_01.tex', mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(lines))

        pass


def get_delta(t_0, t_1):
    delta = [t_1[i]-t_0[i] for i in range(len(t_0))]
    return delta


def parallel_model(graph, od_graph, od_matrix):
    g = graph.copy()
    od_g = od_graph.copy()
    traffic = TrafficModel(g, od_g, od_matrix)
    traffic.run()
    t_k = sum(traffic.get_traveltime())
    flow = sum(traffic.get_flow())
    hours = sum(traffic.get_car_hours())
    distances = sum(traffic.get_car_distances())
    lost_trips = sum(traffic.get_lost_trips().values())
    return t_k, flow, hours, distances, lost_trips

# if __name__ == '__main__':


def main():
    print('***** start *****')
    start = timeit.default_timer()
    model = PpEngine('./results/test_03/')
    model.run('99')
    os.system('cd ./results/test_03/tex/ && pdflatex fig-3.tex')
    stop = timeit.default_timer()
    print('time: ' + str(stop - start))
    print('****** end ******')


def old_main():
    net = read_shp('./data/roads_clean.shp')
    print(nx.info(net))

    states = [(((759564.7, 193863.6), (759486.2, 193874.4)), 0), (((750173.5, 187788.7), (750254.0, 187818.8)), 0), (((750274.52, 187591.83), (750970.7, 188290.6)), 2), (((750205.1, 187513.1), (750274.52, 187591.83)), 2), (((750892.8, 188259.0), (750846.7, 188266.5)), 1), (((750758.67, 185809.8), (750687.71, 185933.99)), 2), (((760146.99, 195565.56), (760061.78, 195228.28)), 2), (((760072.86, 195139.91), (760316.99, 195480.22)), 2), (((761299.9, 197672.1), (760157.04, 195601.1)), 2), (((759276.93, 193205.89), (758864.56, 192928.38)), 2), (((750205.1, 187513.1), (750032.6, 187537.7)), 2), (((757191.29, 190611.49), (757051.9, 190757.8)), 2), (((750029.5, 187527.68), (750111.3, 187316.8)), 2), (((758869.98, 192919.98), (759283.67, 193198.51)), 2), ((
        (759199.86, 192428.21), (759496.12, 193153.65)), 2), (((757295.4, 191208.2), (757357.01, 191276.99)), 1), (((750846.7, 188266.5), (750720.3, 188285.3)), 2), (((760653.9, 196792.7), (761122.2, 197363.1)), 2), (((760395.1, 196965.5), (760902.2, 197368.6)), 2), (((757051.9, 190757.8), (757384.9, 191117.0)), 2), (((756263.0, 190580.5), (756566.7, 190985.3)), 2), (((757048.2, 190761.8), (756692.3, 191010.4)), 2), (((756566.7, 190985.3), (756566.83, 191085.62)), 2), (((759486.67, 194199.2), (759684.8, 194851.49)), 2), (((759684.8, 194851.49), (759921.1, 195903.9)), 2), (((750418.28, 187694.49), (750372.0, 187763.7)), 2), (((756566.83, 191085.62), (757125.6, 191859.3)), 2), (((750183.5, 187097.9), (750024.54, 187597.17)), 2), (((757295.4, 191208.2), (756999.6, 191494.5)), 1)]

    node_map = {}
    for s in states:
        u = s[0][0]
        v = s[0][1]
        node_map[net[u][v]['name'][2:]] = s[0]

    print(node_map)

    new_node_list = [(2052, 0),
                     (2042, 1),
                     (1913, 2),
                     (554, 0),
                     (2131, 0),
                     (1803, 2),
                     (1802, 2),
                     (1706, 2),
                     (460, 2),
                     (332, 2),
                     (461, 2),
                     (562, 2),
                     (1703, 2),
                     (2069, 2),
                     (1095, 2),
                     (1692, 2),
                     (1279, 2),
                     (1907, 2),
                     (1905, 2),
                     (1237, 2),
                     (1276, 2),
                     (1202, 2),
                     (1233, 2),
                     (1798, 2),
                     (1814, 2),
                     (2043, 2),
                     (1498, 2),
                     (471, 2),
                     (1371, 2)
                     ]
    _states = []
    for n, i in new_node_list:
        _states.append((node_map[str(n)], i))

    print(_states)


main()
# =============================================================================
# eof
#
# Local Variables:
# mode: python
# End:
