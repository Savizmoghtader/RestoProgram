from __future__ import print_function
import math
import random
from DiscretePSO import DPSO_Optimizer
import copy

class My_DPSO_Interface(DPSO_Optimizer):

    # pass extra data (the distance dict) into the constructor
    def __init__(self, init_state_str, distance_dict, fdir):
        self.fdir = fdir
        self.distance_dict = distance_dict
        super(My_DPSO_Interface, self).__init__(init_state_str, self.distance_dict, fdir= self.fdir)  # important!

    def move(self):
        """Swaps two cities in the route."""

        initial_energy = self.energy(self.state)

        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

        return self.energy(self.state) - initial_energy

    # def energy(self, state_str):
    #     """Calculates the length of the route."""
    #     e = 0
    #     for i in range(len(state_str)):
    #         e += self.distance_dict[state_str[i-1]][state_str[i]]
    #     return e