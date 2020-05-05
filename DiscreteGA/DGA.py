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


class GeneticAlgorithm(object):

    """Performs GA by calling functions to calculate
    energy and make moves on a state.
    """
    __metaclass__ = abc.ABCMeta

    popSize = 4
    eliteSize = 1
    mutationRate = 0.01
    generations = 3

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
        for i in range(0, self.popSize):
            self.population.append(self.createRoute())

        self.popEnergy = []

    def createRoute(self):
        route = random.sample(self.state, len(self.state)) #E.g. list1 = [1, 2, 3, 4, 5] /print(sample(list1,3)) = [2, 3, 5]
        return route

    @abc.abstractmethod
    def energy(self):
        """Calculate state's energy"""
        pass

    def rankRoutes(self):
        fitnessResults = {}
        for i in range(0, len(self.popEnergy)):
            fitnessResults[i] = self.popEnergy[i]
        popRanked = sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=False)
        return popRanked # sort a list of tuples on the second key.  The first key is the route

    def selection(self, popRanked):
        selectionResults = []
        df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

        for i in range(0, self.eliteSize):
           selectionResults.append(popRanked[i][0])
        for i in range(0, len(popRanked) - self.eliteSize):
            pick = 100 * random.random()
            for i in range(0, len(popRanked)):
                if pick <= df.iat[i, 3]:
                    selectionResults.append(popRanked[i][0])
                    break
        return selectionResults

    def matingPool(self, population, selectionResults):
        matingpool = []
        for i in range(0, len(selectionResults)):
            index = selectionResults[i]
            matingpool.append(population[index])
        return matingpool

    def breed(self, parent1, parent2):
        child = []
        childP1 = []
        childP2 = []

        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))

        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for i in range(startGene, endGene):
            childP1.append(parent1[i])

        childP2 = [item for item in parent2 if item not in childP1] # todo what happens when it is repeated ?

        child = childP1 + childP2
        return child

    def breedPopulation(self, matingpool):
        children = []
        length = len(matingpool) - self.eliteSize
        pool = random.sample(matingpool, len(matingpool))

        for i in range(0, self.eliteSize):
            children.append(matingpool[i])

        for i in range(0, length):
            child = self.breed(pool[i], pool[len(matingpool) - i - 1])
            children.append(child)
        return children


    def mutate(self, individual):
        for swapped in range(len(individual)):
            if (random.random() < self.mutationRate):
                swapWith = int(random.random() * len(individual))

                city1 = individual[swapped]
                city2 = individual[swapWith]

                individual[swapped] = city2
                individual[swapWith] = city1
        return individual

    def mutatePopulation(self, population):
        mutatedPop = []

        for ind in range(0, len(population)):
            mutatedInd = self.mutate(population[ind])
            mutatedPop.append(mutatedInd)
        return mutatedPop

    def nextGeneration(self):
        currentGen = self.population.copy()
        popRanked = self.rankRoutes()
        selectionResults = self.selection(popRanked)
        matingpool = self.matingPool(currentGen, selectionResults)
        children = self.breedPopulation(matingpool)
        nextGeneration = self.mutatePopulation(children)
        return nextGeneration

    def run(self):

        self.popEnergy = self.energy()
        print("Initial energy: " + str(self.rankRoutes()[0][1]))

        progress = []
        progress.append(self.rankRoutes()[0][1])

        for i in range(0, self.generations):
            self.population = self.nextGeneration()
            self.popEnergy = self.energy()
            progress.append(self.rankRoutes()[0][1])

        print("Final distance: " + str(self.rankRoutes()[0][1]))
        bestRouteIndex = self.rankRoutes()[0][0]
        bestRoute = self.population[bestRouteIndex]
        bestRouteEnergy = self.popEnergy[bestRouteIndex]

        plt.plot(progress)
        plt.ylabel('Energy')
        plt.xlabel('Generation')
        plt.show()

        return bestRoute, bestRouteEnergy


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




