from damagemodel import DamageModel
from restorationmodel import RestorationModel
from traffic_fw.trafficmodel import TrafficModel
from traffic_fw.initialize import *
from SimAnneal_Interface import SimAnnealInterface

import random
import ast
import csv

class Engine(object):

    """
    Simulation engine class:
    This Engine is used to find the near optimal restoration programs for transportation networks.

    ----------
    TODO:Needed input Data:
    csv file for restoration names (levels)
    csv file for  capacity losses of objects with respect to each damage level.
    csv file containing the od matrix
    shp files containing the object IDs and their related length, flow direction (one way , two way),  capacity,
    speed limit and damage level.

    ----------
    Attributes/Parameters
    ----------
    test : bool
        a variable to define if the model runs on test data or original data

    capacity_losses : dict
       This attribute defines the different level of capacity losses and the related percentage of capacity loss for
       different objects. Here we have 3 levels of capacity loss for bridges (4th level is the same as the third),
       with 0 , 50% and 100% losses respectively.

    restoration_names : dict
       This attribute shows the different levels for restoration programs.

    self.restoration_types:  TODO: Is this related to different scenarios?

    restoration_constraint : bool
        It shows whether we have constraints in the restoration scenarios or not.

    self.output_directory: directory
       TODO: is this where the output is going to be saved ???

    Methods
    -------
    __init__:
        This is the constructor function of the engine class

    initialize_network:
        This function will initialize the network and will creat the network graph (self.graph) using the information
        in the GIS .shp files and the csv file that contains the od matrix.

    initialize_damage:
        Runs the DamageModel class and "damage" object and gets the damage dictionary (self.damage_dict)
        TODO: Define whether get_damage_dict is directed or not, meaning ...and some elaboration on the process

    initialize_state:
        This function randomly selects objects and is used as an input to the SimAnnealInterface.

    run_traffic_model:
        Runs the TrafficModel class and "traffic" object and gets the travel time (t_k), flow, car hours and distances,
        along with the lost trips.

    run_restoration_model:
        Runs the RestorationModel class and "restoration" object and gets the restoration graphs, times and costs
        respectively.
        TODO: what is sequence=None

    run:
        This function is used to run the Engine class (optimization of the restoration programs) on the object named
        model that was constructed from the Engine class.

    """

    def __init__(self, output_directory='./', bTest = None, bRestConstraint = None, RestTypes = None) :
        self.test = bTest

       # self.capacity_losses = {'Bridge': {0: 0, 1: .5, 2: 1}, 'Road': {0: 0, 1: .7, 2: 1}, 'Tunnel': {0: 0}}
        self.capacity_losses = {}
        with open("capacity_losses.csv", 'r') as data_file:
            data = csv.DictReader(data_file, delimiter=",")
            for row in data:
                item = self.capacity_losses.get(row["Objects"], dict())
                item[int(row["Damage_Level"])] = float(row["Capacity_Loss"])

                self.capacity_losses[row["Objects"]] = item

        #self.restoration_names = {0: 'high priority', 1: 'normal', 2: 'low priority'}
        reader = csv.reader(open('./restoration_names.csv'))
        self.restoration_names = {}
        for row in reader:
            self.restoration_names[int(row[0])] = (row[1])

        self.restoration_types = RestTypes
        self.restoration_constraint = bRestConstraint
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

        self.graph = create_network_graph(self.road_graph, self.od_graph, self.con_edges)
        pass

    def initialize_damage(self):
        self.damage = DamageModel(self.graph, self.capacity_losses)
        self.damage.run()
        self.graph_damaged = self.damage.get_graph()
        self.damage_dict = self.damage.get_damage_dict(directed=False)
        pass

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


    def run_traffic_model(self, graph, od_graph):
        self.traffic = TrafficModel(graph, od_graph, self.od_matrix)
        self.traffic.run()
        t_k = sum(self.traffic.get_traveltime())
        flow = sum(self.traffic.get_flow())
        hours = sum(self.traffic.get_car_hours())
        distances = sum(self.traffic.get_car_distances())
        lost_trips = sum(self.traffic.get_lost_trips().values())
        return t_k, flow, hours, distances, lost_trips

    def run_restoration_model(self, sequence=None):
        self.restoration = RestorationModel(self.graph_damaged)
        self.restoration.run(sequence)
        self.restoration_graphs = self.restoration.get_restoration_graphs()
        self.restoration_times = self.restoration.get_restoration_times()
        self.restoration_costs = self.restoration.get_restoration_costs()
        pass

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