from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import copy
import math
import sys
import time
import random
import signal
import pickle
import datetime
import abc
import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt

class Graph(object):
    def __init__(self, cities, rank: int):
        """
        :param cost_matrix:
        :param rank: rank of the cost matrix
        """
        #self.matrix = cost_matrix
        self.rank = rank
        self.cities = cities
        # noinspection PyUnusedLocal
        self.pheromone = [[1 / (rank * rank) for j in range(rank)] for i in range(rank)]

class AntColony(object):

    """Performs GA by calling functions to calculate
    energy and make moves on a state.
    """
    __metaclass__ = abc.ABCMeta

    """
    :param ant_count:
    :param generations:
    :param alpha: relative importance of pheromone
    :param beta: relative importance of heuristic information
    :param rho: pheromone residual coefficient
    :param q: pheromone intensity
    :param strategy: pheromone update strategy. 0 - ant-cycle, 1 - ant-quality, 2 - ant-density
    """
    Q = 10
    rho = 0.5
    beta = 10.0
    alpha = 1.0
    ant_count = 10
    generations = 100
    update_strategy = 2

    copy_strategy = 'deepcopy'
    user_exit = False
    save_state_on_exit = True

    def __init__(self, initial_state=None, load_state=None, fdir=None):
        if initial_state:
            self.state = self.copy_state(initial_state)
        elif load_state:
            with open(load_state, 'rb') as fh:
                self.state = pickle.load(fh)
        else:
            raise ValueError('No valid values supplied for neither \
             initial_state nor load_state')

        signal.signal(signal.SIGINT, self.set_user_exit)
        self.fdir = fdir

        self.population = []
        # This is initializing a population
        for i in range(0, self.ant_count):
            self.population.append(self.createRoute())

        self.popEnergy = []

    def createRoute(self):
        route = random.sample(self.state, len(self.state)) #E.g. list1 = [1, 2, 3, 4, 5] /print(sample(list1,3)) = [2, 3, 5]
        return route


    @abc.abstractmethod
    def energy(self):
        """Calculate state's energy"""
        pass


    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    def distance(self, city1: dict, city2: dict):
        return math.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)


    def run(self, graph: Graph): #same as solve

        best_cost = float('inf')
        best_solution = []
        for gen in range(self.generations):
            # noinspection PyUnusedLocal
            ants = [_Ant(self, graph) for i in range(self.ant_count)]
            for ant in ants:
                for i in range(graph.rank - 1):
                    ant._select_next()
                # ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                ant.total_cost += self.distance(graph.cities[ant.tabu[-1]], graph.cities[ant.tabu[0]])
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.tabu
                # update pheromone
                ant._update_pheromone_delta()
            self._update_pheromone(graph, ants)
            # print('generation #{}, best cost: {}, path: {}'.format(gen, best_cost, best_solution))
        return best_solution, best_cost


    def save_state(self, fname=None):
        """Saves state"""
        if not fname:
            date = datetime.datetime.now().isoformat().split(".")[0]
            fname = date + "_energy_" + str(self.energy()) + ".state"
        print("Saving state to: %s" % fname)
        with open(self.fdir + 'state.state', "wb") as fh:
            pickle.dump(self.state, fh)


    def set_user_exit(self, signum, frame):
        """Raises the user_exit flag, further iterations are stopped
        """
        self.user_exit = True

    def copy_state(self, state):
        """Returns an exact copy of the provided state
        Implemented according to self.copy_strategy, one of

        * deepcopy : use copy.deepcopy (slow but reliable)
        * slice: use list slices (faster but only works if state is list-like)
        * method: use the state's copy() method
        """
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        elif self.copy_strategy == 'method':
            return state.copy()

    def round_figures(x, n):
        """Returns x rounded to n significant figures."""
        return round(x, int(n - math.ceil(math.log10(abs(x)))))


    def time_string(seconds):
        """Returns time in seconds as a string formatted HHHH:MM:SS."""
        s = int(round(seconds))  # round to nearest second
        h, s = divmod(s, 3600)   # get hours and remainder
        m, s = divmod(s, 60)     # split remainder into minutes and seconds
        return '%4i:%02i:%02i' % (h, m, s)


class _Ant(object):
    def __init__(self, aco: AntColony, graph: Graph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []  # tabu list
        self.pheromone_delta = []  # the local increase of pheromone
        self.allowed = [i for i in range(graph.rank)]  # nodes which are allowed for the next selection


        # self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
        #             range(graph.rank)]  # heuristic information
        self.eta = [[0 if i == j else 1 / self.colony.distance(self.graph.cities[i], self.graph.cities[j]) for j in range(graph.rank)] for i in
                    range(graph.rank)]  # heuristic information
        start = random.randint(0, graph.rank - 1)  # start from any node
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][
                                                                                            i] ** self.colony.beta
        # noinspection PyUnusedLocal
        probabilities = [0 for i in range(self.graph.rank)]  # probabilities for moving to a node in the next step
        for i in range(self.graph.rank):
            try:
                self.allowed.index(i)  # test if allowed list contains i
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
                    self.eta[self.current][i] ** self.colony.beta / denominator
            except ValueError:
                pass  # do nothing
        # select next node by probability roulette
        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        self.allowed.remove(selected)
        self.tabu.append(selected)
        # self.total_cost += self.graph.matrix[self.current][selected]

        self.total_cost += self.colony.distance(self.graph.cities[self.current], self.graph.cities[selected])
        self.current = selected

    # noinspection PyUnusedLocal
    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            if self.colony.update_strategy == 1:  # ant-quality system
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:  # ant-density system
                # noinspection PyTypeChecker
                # self.pheromone_delta[i][j] = self.colony.Q / self.graph.matrix[i][j]
                self.pheromone_delta[i][j] = self.colony.Q / self.colony.distance(self.graph.cities[i], self.graph.cities[j])


            else:  # ant-cycle system
                self.pheromone_delta[i][j] = self.colony.Q / self.total_cost



