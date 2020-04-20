from simanneal import Annealer
from restorationmodel import RestorationModel
from ToolBox import *
from joblib import Parallel, delayed
from DiscretePSO import Swarm

import numpy as np
import random
import csv

class Swarm_Interface(Swarm.SwarmOptimizer):
    """
        Interface to use Swarm package and its classes for Discrete PSO
        Methods
            energy
                Calculates the values for the objective function.
                Example:Calculates the costs of the restoration
    """

    def __init__(self, state, graph, od_graph, od_matrix, graph_damaged, damage, fdir):

        """
        This method is the constructor function of the SimAnnealInterface class and it inherits the __init__ method of
        the anneal class in simanneal package as well.
        """
        # Used to calculate Energy
        self.graph = graph
        self.od_graph = od_graph
        self.od_matrix = od_matrix
        self.graph_damaged = graph_damaged
        self.no_damage = damage[0]
        self.initial_damage = damage[1]
        # Model parameters for indirect costs
        self.mu = np.array([0.94, 0.06])  # % of distribution of cars vs. trucks
        self.xi = np.array([23.02, 130.96])  # value of travel for cars vs. trucks
        self.F_w = np.array([6.7, 33]) / 100  # mean fuel consumption for cars vs. trucks/ 100 km
        self.nu = 1.88  # mean fuel price
        self.rho = np.array([14.39, 32.54]) / 100  # Operating costs (without fuel) for cars vs. trucks/ 100 km
        self.upsilon = 83.27 * 8  # hourly wage [when lost or delayed trips]* 8 hours/day
        self.day_factor = 9  # factor to find the area under the trip distribution curve
        self.fdir = fdir

        with open(self.fdir+'energy.txt', 'w') as f:
            f.write('Energy')

        # self.restoration_types = [0, 1, 2]
        self.restoration_names = {}
        reader = csv.reader(open('./restoration_names.csv'))
        for row in reader:
            self.restoration_names[int(row[0])] = (row[1])
        self.restoration_types = list(self.restoration_names.keys())

        # it inherits the __init__ method of the anneal class in simanneal package
        super(Swarm_Interface, self).__init__(state, graph, od_graph, od_matrix, graph_damaged, damage, fdir=self.fdir)  # important!

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
        """Calculates the length of the route."""
        E = []
        for i in range(self.size_population):
            e = 0
            restoration = RestorationModel(self.graph_damaged)
            restoration.run(self.getState(self.solutions[i].state_idx))
            restoration_graphs = restoration.get_restoration_graphs()
            restoration_times = restoration.get_restoration_times()
            restoration_costs = restoration.get_restoration_costs()
            damaged = []
            damaged.append(get_delta(self.no_damage, self.initial_damage))

            sim_results = Parallel(n_jobs=-2)(delayed(parallel_model)(
                graph, self.od_graph, self.od_matrix) for graph in restoration_graphs[:-1])
            for values in sim_results:
                damaged.append(get_delta(self.no_damage, values))

            for idx, values in enumerate(damaged):
                dt = restoration_times[idx] if idx == 0 else restoration_times[idx] - \
                                                             restoration_times[idx - 1]
                e += sum(restoration_costs[idx]) + dt * (self.day_factor * values[2] * np.sum(self.mu * self.xi) +
                                                         values[3] * np.sum(self.mu * (self.nu * self.F_w + self.rho)) +
                                                         values[4] * self.upsilon)
            self.solutions[i].energy = copy.deepcopy(e)
            E.append(e)

        with open(self.fdir + 'energy_pso.csv', 'a') as f:
            f.write('\n' + str(E[0]))
        return E # list of swarm energies
