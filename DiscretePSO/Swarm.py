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
from restorationmodel import RestorationModel
from ToolBox import *
from joblib import Parallel, delayed
from operator import attrgetter
import matplotlib
import matplotlib.pyplot as plt


def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))


def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)  # get hours and remainder
    m, s = divmod(s, 60)  # split remainder into minutes and seconds
    return '%4i:%02i:%02i' % (h, m, s)

    # This class represent a state
    # each Particle is a state_idx with total energy


class Particle:

    # Create a new state
    def __init__(self, state_idx: [], state_velocity: [], energy=0):
        self.state_idx = state_idx
        self.energy = energy
        self.velocity = state_velocity  # Just for the sake of PSO

    # Compare states
    def __eq__(self, other):
        for i in range(len(self.state_idx)):
            if (self.state_idx[i] != other.state_idex[i]):
                return False
        return True

    # Sort states
    def __lt__(self, other):
        return self.energy < other.energy

    # Print a state
    def __repr__(self):
        return ('({0},{1})\n'.format(self.state_idx, self.energy))

    # Create a shallow copy
    def copy(self):
        return Particle(self.state_idx, self.velocity, self.energy)

    # Create a deep copy
    def deepcopy(self):
        return Particle(copy.deepcopy(self.state_idx), copy.deepcopy(self.velocity), copy.deepcopy(self.energy))

    def clearVelocity(self):
        del self.velocity[:]

    def setVelocity(self, new_velocity):
        self.velocity = new_velocity

    # Update energy
    def update_energy(self, energy):
        # Reset energy
        self.energy = energy


class SwarmOptimizer(object):
    """Performs simulated annealing by calling functions to calculate
    energy and make moves on a state.  The temperature schedule for
    annealing may be provided manually or estimated automatically.
    """
    __metaclass__ = abc.ABCMeta

    n_random = 4
    size_population = 4
    nswarm = 2
    c1 = 1.5  # Individual coeff
    c2 = 1.3  # Social coeff
    maxIterations = 20
    w = 0.9  # Inertial Coeff

    copy_strategy = 'deepcopy'
    user_exit = False
    save_state_on_exit = False

    def __init__(self, init_state, graph, od_graph, od_matrix, graph_damaged, damage, load_state=None, fdir=None):

        self.fdir = fdir
        self.AllNodes = init_state
        self.node_indices = list(range(len(init_state)))
        self.dict_idx = {}
        for i in range(len(self.AllNodes)):
            self.dict_idx[self.AllNodes[i]] = self.node_indices[i]

        if init_state:
            self.init_state = self.copy_state(init_state)
        elif load_state:
            with open(load_state, 'rb') as fh:
                self.init_state = pickle.load(fh)
        else:
            raise ValueError('No valid values supplied for neither \
            initial_state nor load_state')

        self.init_state_idx = self.getStateIDX(init_state)
        signal.signal(signal.SIGINT, self.set_user_exit)
        self.solutions = self.get_random_swarm(Use_Swap=False)  # initialize swarm with random solutions and the init_state
        self.E_swarm = self.energy()  # calculate energy of the swarm and update all solutions in the swarm

        # initialize gbest and pbest
        self.pbest = copy.deepcopy(self.solutions)
        gbest_temp = copy.deepcopy(self.solutions)
        gbest_temp.sort()
        self.gbest = gbest_temp[0].deepcopy()  # best solution (MIN)

    def save_state(self, fname=None):
        """Saves state"""
        gbest_state = self.getState(self.gbest.state_idx)
        gbest_energy = self.gbest.energy
        if not fname:
            date = datetime.datetime.now().isoformat().split(".")[0]
            fname = date + "_energy_" + str(gbest_energy) + ".state"
        print("Saving state to: %s" % fname)
        with open(self.fdir + 'state.state', "wb") as fh:
            pickle.dump(gbest_state, fh)

    @abc.abstractmethod
    def move(self):
        """Create a state change"""
        return

    @abc.abstractmethod
    def energy(self):
        pass

    # Get the indices (routes) of a state
    def getStateIDX(self, state):
        state_idx = copy.deepcopy(self.node_indices)
        for i in range(len(state)):
            state_idx[i] = self.dict_idx[state[i]]
        return state_idx

    # Get a state using a provided index
    def getState(self, state_idx):
        # get city names from indices
        temp = list(self.dict_idx.keys())
        state = []
        for i in state_idx:
            state.append(temp[i])
        return state

    # Get the best random solution (indices)
    def get_random_swarm(self, Use_Swap: bool = False):
        bufferIdx = self.init_state_idx.copy()
        length = len(self.init_state_idx) - 1

        if Use_Swap:
            tempVel = []
            for N in range(random.randint(0, length)):  # random initial velocities
                tempVel.append((random.randint(0, length - 1), random.randint(0, length - 1), self.w))
        else:
            tempVel = [0] * (length + 1)

        solutions = [Particle(bufferIdx, tempVel)]

        for i in range(self.size_population - 1):
            bufferIdx = self.init_state_idx.copy()
            # bufferIdx.pop(self.init_state_idx[0])
            random.shuffle(bufferIdx)
            # bufferIdx.insert(0, self.init_state_idx[0])
            if Use_Swap:
                tempVel = []
                for N in range(random.randint(0, length)):  # random initial velocities
                    tempVel.append((random.randint(0, length - 1), random.randint(0, length - 1), self.w))
            else:
                tempVel = [0] * (length + 1)

            particle = Particle(bufferIdx, tempVel)
            solutions.append(particle)

        return solutions


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

    def update(self, step, gbestE, acceptance, improvement, highlight=False):

        if highlight:
            cl = "\033[;1m" + "\033[0;32m"  # Yellow
        else:
            cl = "\033[1;31m"

        elapsed = time.time() - self.start
        if step == 0:
            print(cl + 'GBest Energy    Accept   Improve     Elapsed   Remaining')
            sys.stdout.write('\n%12.2f                      %s            ' % \
                             (gbestE, time_string(elapsed)))
            sys.stdout.flush()
            # with open(self.fdir+'log.csv','w') as f:
            #     f.write('GBest Energy,Accept,Improve,Elapsed,Remaining')
            #     f.write('\n'+','.join([str(gbestE), time_string(elapsed)]))

        else:
            remain = (self.maxIterations - step) * (elapsed / step)
            sys.stdout.write(cl + '\n%12.2f  %7.2f%%  %7.2f%%  %s  %s' % \
                             (gbestE, 100.0 * acceptance, 100.0 * improvement, \
                              time_string(elapsed), time_string(remain))),
            sys.stdout.flush()
            # with open(self.fdir+'log.csv','a') as f:
            #     f.write('\n'+','.join([str(gbestE),str(acceptance),str(improvement),time_string(elapsed), time_string(remain)]))


    def DPSO(self):
        self.start = time.time()
        step, trials, accepts, improves = 0, 0, 0, 0
        length_state = len(self.init_state_idx) - 1
        fig, ax = plt.subplots()
        ax.grid()

        delta_w = (self.w - 0.4) / (self.maxIterations - 1)
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)

        # for each time step iteration
        for t in range(self.maxIterations):
            step += 1
            # for each particle in the swarm
            for i in range(self.size_population):
                trials += 1

                previous_route = np.array(self.solutions[i].state_idx)
                inertia = self.w * np.array(self.solutions[i].velocity)
                personal = self.c1 * r1 * (np.array(self.pbest[i].state_idx) - previous_route)
                social = self.c2 * r2 * (np.array(self.gbest.state_idx) - previous_route)

                new_velocity = inertia + personal + social

                temp_route = np.add(previous_route, new_velocity)
                _, _, new_route = np.unique(temp_route, return_index=True, return_inverse=True, return_counts=False)

                self.solutions[i].setVelocity(new_velocity.tolist())
                self.solutions[i].state_idx = new_route.tolist()

            # gets cost of the current solution
            E = self.energy()

            for i in range(self.size_population):
                if self.solutions[i].energy < self.pbest[i].energy:  # checks if current solution is pbest solution
                    self.pbest[i] = self.solutions[i].deepcopy()  # copy of the pbest solution
                    accepts += 1
                ax.scatter(t, self.solutions[i].energy)
                self.update(step, self.pbest[i].energy, accepts / trials, improves / trials, False)

            for i in range(self.size_population):
                if self.pbest[i].energy < self.gbest.energy:
                    self.gbest = self.pbest[i].deepcopy()  # copy of the pbest solution
                    improves += 1

            self.update(step, self.gbest.energy, accepts / trials, improves / trials, True)
            self.w = self.w - delta_w

            #trials, accepts, improves = 0, 1, 1
        ax.set(xlabel='Iteration', ylabel='Cost', title='PSO for TSP')
        plt.show()
        if self.save_state_on_exit:
            self.save_state()

        return self.getState(self.gbest.state_idx), self.gbest.energy


    def DPSO_Swap(self):

        self.start = time.time()
        step, trials, accepts, improves = 0, 0, 0, 0
        length_state = len(self.init_state_idx) - 1
        fig, ax = plt.subplots()
        ax.grid()

        # for each time step iteration
        for t in range(self.maxIterations):
            step += 1
            # for each particle in the swarm
            for i in range(self.size_population):
                trials += 1
                temp_velocity = self.solutions[i].velocity

                # generates all swap operators to calculate (pbest - x(t-1))
                for n in range(length_state):
                    if self.solutions[i].state_idx[n] != self.pbest[i].state_idx[n]:
                        # generates swap operator
                        swap_operator = (n, self.pbest[i].state_idx.index(self.solutions[i].state_idx[n]), self.c1)
                        # append swap operator in the list of velocity
                        temp_velocity.append(swap_operator)

                # generates all swap operators to calculate (gbest - x(t-1))
                for n in range(length_state):
                    if self.solutions[i].state_idx[n] != self.gbest.state_idx[n]:
                        swap_operator = (n, self.gbest.state_idx.index(self.solutions[i].state_idx[n]), self.c2)
                        temp_velocity.append(swap_operator)

                # updates velocity: V(t) = w.V(t-1) + (pbest - x(t-1)) + (gbest - x(t-1))
                self.solutions[i].setVelocity(temp_velocity)

                # generates new solution for particle: X(t) = X(t-1) + V(t)
                for swap_operator in temp_velocity:
                    if random.random() <= swap_operator[2]:
                        aux = self.solutions[i].state_idx[swap_operator[0]]
                        self.solutions[i].state_idx[swap_operator[0]] = self.solutions[i].state_idx[swap_operator[1]]
                        self.solutions[i].state_idx[swap_operator[1]] = aux
                #print("Solution[{0}] Energy = {1} ".format(i, self.solutions[i].energy))

            E = self.energy()

            for i in range(self.size_population):
                if self.solutions[i].energy < self.pbest[i].energy:  # checks if current solution is pbest solution
                    self.pbest[i] = self.solutions[i].deepcopy()  # copy of the pbest solution
                    improves += 1
                ax.scatter(t, self.solutions[i].energy)
                self.update(step, self.pbest[i].energy, accepts / trials, improves / trials, False)

            for i in range(self.size_population):
                if self.pbest[i].energy < self.gbest.energy:
                    self.gbest = self.pbest[i].deepcopy()  # copy of the pbest solution
                    accepts += 1

                self.update(step, self.gbest.energy, accepts / trials, improves / trials, True)
                #print("   -->  gbest.state_idx = ", self.gbest.state_idx)

            trials, accepts, improves = 0, 1, 1
            #print("\ngbest Energy = {0} ".format(self.gbest.energy))

        ax.set(xlabel='Iteration', ylabel='Cost', title='PSO for TSP')
        plt.show()
        if self.save_state_on_exit:
            self.save_state()

        return self.getState(self.gbest.state_idx), self.gbest.energy
