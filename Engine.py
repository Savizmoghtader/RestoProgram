from damagemodel import DamageModel
from restorationmodel import RestorationModel
from traffic_fw.trafficmodel import TrafficModel
from traffic_fw.initialize import *
from SimAnneal_Interface import SimAnnealInterface

import random
import ast


class Engine(object):

    """
    Simulation engine class:

    ...

    Attributes/Parameters
    ----------
    test : bool
        a variable to define if the model runs on test data or original data
    capacity_losses : dict
        this includes ???

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(self, output_directory='./'):
        self.test =  True  #False #
        self.capacity_losses = {'Bridge': {0: 0, 1: .5, 2: 1, 3: 1}, 'Road': {
            0: 0, 1: .7, 2: 1, 3: 1}, 'Tunnel': {0: 0}}

        # TODO: Load data from csv file
        self.restoration_names = {
            0: 'high priority', 1: 'normal', 2: 'low priority'}
        self.restoration_types = [0, 1, 2]

        self.restoration_constraint = False
        self.output_directory = output_directory

    def initialize_network(self):
        if self.test:
            self.road_graph = read_shp('./test_data/roads.shp')
            self.od_graph = create_od_graph('./test_data/centroids.shp')
            self.con_edges = read_shp('./test_data/connections.shp')
            self.od_matrix = np.genfromtxt('./test_data/od.csv', delimiter=',')
        else:
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
        self.restoration = RestorationModel(self.graph_damaged)
        self.restoration.run(sequence)
        self.restoration_graphs = self.restoration.get_restoration_graphs()
        self.restoration_times = self.restoration.get_restoration_times()
        self.restoration_costs = self.restoration.get_restoration_costs()
        pass

    def run_traffic_model(self, graph, od_graph):
        # set up traffic model
        self.traffic = TrafficModel(graph, od_graph, self.od_matrix)
        # run traffic simulation
        self.traffic.run()
        t_k = sum(self.traffic.get_traveltime())
        flow = sum(self.traffic.get_flow())
        hours = sum(self.traffic.get_car_hours())
        distances = sum(self.traffic.get_car_distances())
        lost_trips = sum(self.traffic.get_lost_trips().values())
        return t_k, flow, hours, distances, lost_trips

    def initialize_state(self):
        init_edges = list(self.damage_dict.keys())
        random.shuffle(init_edges)

        init_state = []

        for edge in init_edges:
            if self.restoration_constraint:
                init_state.append((edge, 0))
            else:
                init_state.append((edge, random.choice(self.restoration_types)))
        return init_state

    def load_state(self, filename):
        with open(filename, 'r') as f:
            state = ast.literal_eval(f.read())
        return state

    def load_damage(self, filename):
        with open(filename, 'r') as f:
            damaged = ast.literal_eval(f.read())
        return damaged

    def run(self):
        self.initialize_network()
        self.initialize_damage()
        init_state = self.initialize_state()

        no_damage = self.run_traffic_model(self.graph, self.od_graph)
        initial_damage = self.run_traffic_model(
            self.graph_damaged.copy(), self.od_graph.copy())

        damage = [no_damage, initial_damage]

        optimize = SimAnnealInterface(init_state, self.graph, self.od_graph,
                               self.od_matrix, self.graph_damaged, damage, self.output_directory)

        optimize.copy_strategy = "slice"

        state, e = optimize.anneal()

        print("consequences: %i" % e)

        self.restoration_results = RestorationModel(self.graph_damaged)
        # (object,schedule time,needed time,#resources,intervention type, assignd resource)
        sequence_formated = self.restoration_results.format(state)

        with open(self.output_directory+'state.txt', 'w') as f:
            f.write(str(state))

        with open(self.output_directory+'state_formated.txt', 'w') as f:
            f.write(str(sequence_formated))
        pass