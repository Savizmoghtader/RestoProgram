from simanneal import Annealer
from restorationmodel import RestorationModel
from ToolBox import *
from joblib import Parallel, delayed


import numpy as np
import random


class SimAnnealInterface(Annealer):
    """
        Interface to use simaneal package and its classes

        ...

        Attributes/Parameters
        ----------
        test : bool
            a variable to define if the model runs on test data or original data
        capacity_losses : dict
            this includes ???

        Methods/Functions
        -------
        says(sound=None)
            Prints the animals name and what sound it makes
        """

    def __init__(self, state, graph, od_graph, od_matrix, graph_damaged, damage, fdir):
        self.graph = graph
        self.od_graph = od_graph
        self.od_matrix = od_matrix
        self.graph_damaged = graph_damaged
        self.no_damage = damage[0]
        self.initial_damage = damage[1]
        self.fdir = fdir

        # Model parameters for indirect costs
        self.mu = np.array([0.94, 0.06])
        self.xi = np.array([23.02, 130.96])
        self.F_w = np.array([6.7, 33])/100
        self.nu = 1.88
        self.rho = np.array([14.39, 32.54])/100
        self.upsilon = 83.27 * 8
        self.day_factor = 9

        with open(self.fdir+'energy.txt', 'w') as f:
            f.write('Energy')

        self.restoration_types = [0, 1, 2]
        super(SimAnnealInterface, self).__init__(state, fdir=self.fdir)  # important!

    def move(self):
        """Swaps two object in the restoration schedual."""
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

        return e
