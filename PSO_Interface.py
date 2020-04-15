from simanneal import Annealer
from restorationmodel import RestorationModel
from ToolBox import *
from joblib import Parallel, delayed
from DiscretePSO import DPSO_Optimizer

import numpy as np
import random
import csv

class My_DPSO_Interface(DPSO_Optimizer):
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
        self.fdir = fdir

        # with open(self.fdir+'energy.txt', 'w') as f:
        #     f.write('Energy')

        # self.restoration_types = [0, 1, 2]
        self.restoration_names = {}
        reader = csv.reader(open('./restoration_names.csv'))
        for row in reader:
            self.restoration_names[int(row[0])] = (row[1])
        self.restoration_types = list(self.restoration_names.keys())

        # it inherits the __init__ method of the anneal class in simanneal package
        super(My_DPSO_Interface, self).__init__(state, graph, od_graph, od_matrix, graph_damaged, damage, fdir=self.fdir)  # important!

    def move(self):
        """Swaps two object in the restoration schedual."""
        # random.randint(a, b): Return a random integer N such that a <= N <= b.

        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

        # change type of restoration for one state
        c = random.choice(self.restoration_types)
        self.state[a] = (self.state[a][0], c)

    # def energy(self):
    #     """Calculates the costs of the restoration."""
    #     return
