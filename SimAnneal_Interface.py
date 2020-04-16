from simanneal import Annealer
from restorationmodel import RestorationModel
from ToolBox import *
from joblib import Parallel, delayed


import numpy as np
import random
import csv

class SimAnnealInterface(Annealer):
    """
        Interface to use simaneal package and its classes

        Methods
        ...
            __init__
               the constructor function of the class it requires the following input parameters:

                state
                graph
                od_graph
                od_matrix
                graph_damaged
                damage
                fdir

                Example: __init__(init_state, self.graph, self.od_graph,
                self.od_matrix, self.graph_damaged, damage, self.output_directory)

            move
                 Swaps two object in the restoration schedule.

            energy
                Calculates the values for the objective function.
                Example:Calculates the costs of the restoration
    """

    def __init__(self, state, graph, od_graph, od_matrix, graph_damaged, damage, fdir):

        """
        This method is the constructor function of the SimAnnealInterface class and it inherits the __init__ method of
        the anneal class in simanneal package as well.
        """

        self.graph = graph
        self.od_graph = od_graph
        self.od_matrix = od_matrix
        self.graph_damaged = graph_damaged
        self.no_damage = damage[0]
        self.initial_damage = damage[1]
        self.fdir = fdir

        # Model parameters for indirect costs
        self.mu = np.array([0.94, 0.06])                # % of distribution of cars vs. trucks
        self.xi = np.array([23.02, 130.96])             # value of travel for cars vs. trucks
        self.F_w = np.array([6.7, 33])/100              # mean fuel consumption for cars vs. trucks/ 100 km
        self.nu = 1.88                                  # mean fuel price
        self.rho = np.array([14.39, 32.54])/100         # Operating costs (without fuel) for cars vs. trucks/ 100 km
        self.upsilon = 83.27 * 8                        # hourly wage [when lost or delayed trips]* 8 hours/day
        self.day_factor = 9                             # factor to find the area under the trip distribution curve(average value*9= total trips per day for a zone)

        with open(self.fdir+'energy.txt', 'w') as f:
            f.write('Energy')

        # self.restoration_types = [0, 1, 2]
        self.restoration_names = {}
        reader = csv.reader(open('./restoration_names.csv'))
        for row in reader:
            self.restoration_names[int(row[0])] = (row[1])
        self.restoration_types = list(self.restoration_names.keys())

        # it inherits the __init__ method of the anneal class in simanneal package
        super(SimAnnealInterface, self).__init__(state, fdir=self.fdir)  # important!

    def move(self):
        """Swaps two object in the restoration schedual."""
        # random.randint(a, b): Return a random integer N such that a <= N <= b.

        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

        # change type of restoration for one state
        c = random.choice(self.restoration_types)
        self.state[a] = (self.state[a][0], c)

    def energy(self):
        """Calculates the costs of the restoration."""
        e = 0

        restoration = RestorationModel(self.graph_damaged)
        restoration.run(self.state)
        restoration_graphs = restoration.get_restoration_graphs()
        restoration_times = restoration.get_restoration_times()
        restoration_costs = restoration.get_restoration_costs()

        damaged = []
        damaged.append(get_delta(self.no_damage, self.initial_damage))

        sim_results = Parallel(n_jobs=4)(delayed(parallel_model)(
            graph, self.od_graph, self.od_matrix) for graph in restoration_graphs[:-1])
        for values in sim_results:
            damaged.append(get_delta(self.no_damage, values))

        for idx, values in enumerate(damaged):
            dt = restoration_times[idx] if idx == 0 else restoration_times[idx] - \
                restoration_times[idx-1]
            e += sum(restoration_costs[idx]) + dt * (self.day_factor * values[2] * np.sum(self.mu*self.xi) +
                                                     values[3] * np.sum(self.mu * (self.nu * self.F_w + self.rho)) + values[4] * self.upsilon)
        with open(self.fdir+'energy.csv', 'a') as f:
            f.write('\n'+str(e))

        #print("SA Energy Functions is Uesd ....")

        return e
