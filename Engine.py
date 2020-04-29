from damagemodel import DamageModel
from restorationmodel import RestorationModel
from traffic_fw.trafficmodel import TrafficModel
from traffic_fw.initialize import *
from SimAnneal_Interface import SimAnnealInterface
from PSO_Interface import My_PSO_Interface
from Swarm_Interface import Swarm_Interface
from DGA_Interface import GAInterface

import random
import ast
import csv
import os

class Engine(object):

    """
    Simulation engine class:
    This Engine is used to find the near optimal restoration programs for transportation networks.

    Needed input Data
    ----------

        csv file for restoration names (levels)
        csv file for capacity losses of objects with respect to each damage level.
        csv file containing the od matrix
        shape files containing the object IDs and their related length, width (determined from the object class),
        flow direction (one way: true or false), capacity, speed limit and damage level.

    Parameters
    ----------
        self.test : bool
            a variable to define if the model runs on test data or original data.

        self.capacity_losses : Nested dict
            This attribute defines the different level of capacity losses and the related percentage of capacity loss
            for different objects. The data is imported from a csv file.
            In this example, the objects 'Bridge' , 'Road', and 'Tunnel' are the keys; the nested keys refer to the
            capacity loss levels 0, 1 and 2 and the values relate to the percentage of capacity loss
            (e.g 0 , 50% and 100% for bridges).

            self.capacity_losses = {'Bridge': {0: 0, 1: .5, 2: 1}, 'Road': {0: 0, 1: .7, 2: 1}, 'Tunnel': {0: 0}}

        self.restoration_names : dict
            This attribute shows the different levels for restoration programs. The data is imported from a csv file
            In this example :
            self.restoration_names = {0: 'high priority', 1: 'normal', 2: 'low priority'}

        self.restoration_types:  list
            A list of the the keys in the self.restoration_names referring to the intervention level

        self.restoration_constraint : bool
            It shows whether we have constraints in the restoration scenarios or not.

        self.output_directory: directory path
            It shows where the output is going to be saved.

    Methods
    -------
        __init__:
            This is the constructor function of the engine class

        initialize_network:
            This function will initialize the network and will creat the network graph (self.graph) using the
            information in the GIS .shp files and the csv file that contains the od matrix.

        initialize_damage:
            Runs the DamageModel class on the "damage" object and creates the damage dictionary (self.damage_dict)

        initialize_state:
            This function randomly selects objects and is used as an input to the SimAnnealInterface.

        run_traffic_model:
            Runs the TrafficModel class and "traffic" object and gets the travel time (t_k), flow, car hours and
            distances, along with the lost trips.

        run:
            This function is used to run the Engine class (optimization of the restoration programs) on the object named
            model that was constructed from the Engine class.
    """

    def __init__(self, output_directory='./', bTest = None, bRestConstraint = None) :

        """
        This is the constructor function of the engine class
        
        Input parameters
        ----------
            output_directory: directory path
                It shows where the output is going to be saved.
                
            bTest : bool
                a variable to define if the model runs on test data or original data.
                
            bRestConstraint : bool
                It shows whether we have constraints in the restoration scenarios or not.
        """
        self.test = bTest

        self.capacity_losses = {}
        data = csv.DictReader(open("capacity_losses.csv", 'r'), delimiter=",")
        for row in data:
            item = self.capacity_losses.get(row["Objects"], dict())
            item[int(row["Damage_Level"])] = float(row["Capacity_Loss"])
            self.capacity_losses[row["Objects"]] = item

        self.restoration_names = {}
        reader = csv.reader(open('./restoration_names.csv'))
        for row in reader:
            self.restoration_names[int(row[0])] = (row[1])

        self.restoration_types = list(self.restoration_names.keys()) #TODO: check to make sure
        self.restoration_constraint = bRestConstraint
        self.output_directory = output_directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    def initialize_network(self):

        """
        This function will initialize the network and will creat the network graph (self.graph) using the information
        in the GIS .shp files and the csv file that contains the od matrix.
        If the test is true the function will use the test data while when test is false the function will run on the
        actual data. The output of the function is the network graph.

        Input data:
        shape file (.shp) for the roads, centroids and connections.
        csv file containing the od matrix
        """

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
            #
            # self.road_graph = read_shp('./data_Newversion/critical_objs_removed.shp')
            # self.od_graph = create_od_graph('./data_Newversion/centroids.shp')
            # self.con_edges = read_shp('./data_Newversion/connections.shp')
            # self.od_matrix = np.genfromtxt('./data_Newversion/od.csv', delimiter=',')

        self.graph = create_network_graph(self.road_graph, self.od_graph, self.con_edges)

        pass


    def initialize_damage(self):

        """
         Runs the DamageModel class on "damage" object and creates the damage dictionary (self.damage_dict)

        Input parameters:
        ----------
            self.graph: DiGraph
                A  directed graph constructed using the create_network_graph () method in initialize network
                method.

            self.capacity_losses: Nested dict
                This attribute defines the different level of capacity losses and the related percentage of capacity loss
                for different objects. The data is imported from a csv file.
        """

        self.damage = DamageModel(self.graph, self.capacity_losses)
        self.damage.run()
        self.graph_damaged = self.damage.get_graph()
        self.damage_dict = self.damage.get_damage_dict(directed=False)
        pass

        # self.damage_new_saviz1 = {((750173.5, 187788.7), (750254.0, 187818.8)): ['a-2042', 0],
        #                       ((759564.7, 193863.6), (759486.2, 193874.4)): ['b-2052', 0],
        #                       ((757051.9, 190757.8), (757384.9, 191117.0)): ['b-1237', 0],
        #                       ((761299.9, 197672.1), (760157.04, 195601.1)): ['a-1913', 1]}

    def initialize_state(self):

        """
        This function randomly selects objects and returns init_state (to be used as an input to the SimAnnealInterface.)
        """
        # init_edges = list(self.damage_new_saviz1.keys()) # gets the edges of damaged objects
        init_edges = list(self.damage_dict.keys()) # gets the edges of damaged objects
        random.shuffle(init_edges)

        init_state = []
        for edge in init_edges:
            if self.restoration_constraint:  #  we are assigning high priority restoration to all damaged edges (it would reach to optimal results faster)
                init_state.append((edge, 0))
            else:
                init_state.append((edge, random.choice(self.restoration_types)))
        return init_state


    def run_traffic_model(self, graph, od_graph):

        """
        Runs the TrafficModel class on "traffic" object and gets the travel time (t_k), flow, car hours and distances,
        along with the values for lost trips dict.
        """

        self.traffic = TrafficModel(graph, od_graph, self.od_matrix)
        self.traffic.run()
        t_k = sum(self.traffic.get_traveltime())
        flow = sum(self.traffic.get_flow())
        hours = sum(self.traffic.get_car_hours())
        distances = sum(self.traffic.get_car_distances())
        lost_trips = sum(self.traffic.get_lost_trips().values())
        return t_k, flow, hours, distances, lost_trips


    def run(self):

        """
        This function is used to run the Engine class (optimization of the restoration programs) on the object
        that was constructed from the Engine class.

        This is the main function of the Engine class that includes the following methods, and classes:

            methods:
            -------
                initialize_network
                initialize_damage
                initialize_state
                run_traffic_model
                anneal
                format

            Class:
            -------
                SimAnnealInterface(init_state, self.graph, self.od_graph,self.od_matrix, self.graph_damaged, damage,
                self.output_directory)

                RestorationModel(self.graph_damaged)

        """
        self.initialize_network()
        self.initialize_damage()
        init_state = self.initialize_state()

        # init_state = [(((6000.0, 4000.0), (6000.0, 10000.0)), 0), (((6000.0, 10000.0), (12000.0, 18000.0)), 0),
        #  (((6000.0, 10000.0), (0.0, 18000.0)), 2), (((3000.0, 0.0), (0.0, 18000.0)), 2)]

        # getting t_k, flow, hours, distances, lost_trips by running the traffic model before damage
        no_damage = self.run_traffic_model(self.graph, self.od_graph)

        # getting t_k, flow, hours, distances, lost_trips by running the traffic model after damage
        initial_damage = self.run_traffic_model( self.graph_damaged.copy(), self.od_graph.copy())


        damage = [no_damage, initial_damage]


        ######################## PSO OPTIMIZATION ##########################
        # optimize = My_PSO_Interface(init_state, self.graph, self.od_graph,
        #                        self.od_matrix, self.graph_damaged, damage, self.output_directory)
        # optimize.copy_strategy = "slice"
        # state, e = optimize.DPSO()
        ###################################################################

        # ##################### PSO OPTIMIZATION - SWARM ####################
        # optimize = Swarm_Interface(init_state, self.graph, self.od_graph,
        #                       self.od_matrix, self.graph_damaged, damage, self.output_directory)
        # optimize.copy_strategy = "slice"
        # state, e = optimize.DPSO()
        # ###################################################################

        ##################### SIMANNEAL OPTIMIZATION ######################
        optimize = SimAnnealInterface(init_state, self.graph, self.od_graph,
                               self.od_matrix, self.graph_damaged, damage, self.output_directory)
        optimize.copy_strategy = "slice"
        state, e = optimize.anneal()  # Minimizes the energy of a system by simulated annealing
        ##################################################################

        # #######################  GENETIC ALGORITHM OPTIMIZATION ##################
        # optimize = GAInterface(init_state, self.graph, self.od_graph,
        #                               self.od_matrix, self.graph_damaged, damage, self.output_directory)
        # optimize.copy_strategy = "slice"
        # state, e = optimize.anneal()  # Minimizes the energy of a system by simulated annealing
        #
        # #############################################################################


        print("consequences: %i" % e)

        self.restoration_results = RestorationModel(self.graph_damaged)


        # (object,schedule time (when it's finished),needed time,# of resources,intervention type, assignd resource)
        sequence_formated = self.restoration_results.format(state)

        with open(self.output_directory+'state.txt', 'w') as f:
            f.write(str(state))

        with open(self.output_directory+'state_formated.txt', 'w') as f:
            f.write(str(sequence_formated))
        pass