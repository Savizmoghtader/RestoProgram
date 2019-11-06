from traffic_fw.trafficmodel import TrafficModel

import collections
import copy
import numpy as np


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


def merge_dicts(dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def sort_dic(d):
    l = []
    for k,v in d.items():
        l.append('_'.join(k.split("_")[:-1])+'_')
    c=collections.Counter(l)
    x = [i for i in c if c[i]>1]
    t = []
    for j in x:
        e = []
        for i in range(1,c[j]+1):
            e.append(j+str(i))
        t.append(e)
    return t

def get_edge_attribute(graph,edge,attribute):
    try:
        return graph[edge[0]][edge[1]][attribute]
    except:
        return None

def set_edge_attribute(graph,edge,attribute,value):
    graph[edge[0]][edge[1]][attribute] = value
    pass

def remove_duplicates(seq):
    """removes duplicates from a list"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_small_indices(list,x):
    """return indices of the x smallest values of a list """
    max_value = max(list)
    l = copy.copy(list)
    idx_list = []
    for i in range(x):
        idx = l.index(min(l))
        idx_list.append(idx)
        l[idx] = l[idx] + max_value + 1
    return idx_list

def get_resources_array(resources_matrix,duration,resources):
    """return array which can be assignd to a restoration task"""
    resources_array = None
    for j in range(resources_matrix.shape[1]):
        if np.sum(resources_matrix,axis=0)[j] >= resources:
            i_idx = []
            for i in range(resources_matrix.shape[0]):
                if resources_matrix[i,j] == 1:
                    i_idx.append(i)

            i_ok = []
            for idx in i_idx:
                if np.sum(resources_matrix[idx,j:j+duration]) >= duration:
                    i_ok.append(idx)

            if len(i_ok) >= resources:
                i_ok.sort()
                resources_array = (i_ok[0:resources],j)
                break
    return resources_array

