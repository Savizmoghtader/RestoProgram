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
    h, s = divmod(s, 3600)   # get hours and remainder
    m, s = divmod(s, 60)     # split remainder into minutes and seconds
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

class DPSO_Optimizer(object):

    """Performs simulated annealing by calling functions to calculate
    energy and make moves on a state.  The temperature schedule for
    annealing may be provided manually or estimated automatically.
    """
    __metaclass__ = abc.ABCMeta

    n_random = 2
    size_population = 2
    nswarm = 2
    c1 = 1.5 # Individual coeff
    c2 = 1.3 # Social coeff
    maxIterations = 10
    w = 0.9 # Inertial Coeff

    copy_strategy = 'deepcopy'
    user_exit = False
    save_state_on_exit = False

    def __init__(self, init_state, graph, od_graph, od_matrix, graph_damaged, damage, load_state=None, fdir=None):

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
        self.nNodes = len(init_state)
        self.AllNodes = init_state
        self.node_indices = list(range(self.nNodes))
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

        self.init_state_idx = self.getStateIDX(self.init_state)
        self.home_idx = self.init_state_idx[0]

        signal.signal(signal.SIGINT, self.set_user_exit)

    def save_state(self, fname=None):
        """Saves state"""
        gbest_state = self.getState(self.gbest.state_idx)
        gbest_energy = self.gbest.energy
        if not fname:
            date = datetime.datetime.now().isoformat().split(".")[0]
            fname = date + "_energy_" + str(gbest_energy) + ".state"
        print("Saving state to: %s" % fname)
        with open(self.fdir+'state.state', "wb") as fh:
            pickle.dump(gbest_state, fh)

    @abc.abstractmethod
    def move(self):
        """Create a state change"""
        return

    
    def energy(self, state):
        """Calculates the length of the route."""
        e = 0
        restoration = RestorationModel(self.graph_damaged)
        restoration.run(state)
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
        with open(self.fdir + 'energy.csv', 'a') as f:
            f.write('\n' + str(e))
        # print("     ------>  PSO Energy Functions is Used ....")

        return e


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
    def get_random_solution(self, use_weights: bool = False):
        citiesIdx = self.node_indices.copy()
        #citiesIdx.pop(self.home_idx)
        random.shuffle(citiesIdx)
        #citiesIdx.insert(0, self.home_idx)
        cities_str = self.getState(citiesIdx)
        E = self.energy(self.getState(citiesIdx))
        rand_state_idx = citiesIdx.copy()

        for i in range(self.n_random - 1):
            citiesIdx = self.node_indices.copy()
            citiesIdx.pop(self.home_idx)
            # Shuffle cities at random
            random.shuffle(citiesIdx)
            citiesIdx.insert(0, self.home_idx)
            E_new = self.energy(self.getState(citiesIdx))
            if (E_new < E):
                E = copy.deepcopy(E_new)
                # print('Energy = ', E)
                rand_state_idx = citiesIdx.copy()
        # Return the best solution amongst all randoms ( Energy)
        return rand_state_idx

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
            cl = "\033[;1m" + "\033[0;32m"  #Yellow
        else:
            cl = "\033[1;31m"

        elapsed = time.time() - self.start
        if step == 0:
            print(cl+'GBest Energy    Accept   Improve     Elapsed   Remaining')
            sys.stdout.write('\n%12.2f                      %s            ' % \
                (gbestE, time_string(elapsed)))
            sys.stdout.flush()
            # with open(self.fdir+'log.csv','w') as f:
            #     f.write('GBest Energy,Accept,Improve,Elapsed,Remaining')
            #     f.write('\n'+','.join([str(gbestE), time_string(elapsed)]))

        else:
            remain = (self.maxIterations - step) * (elapsed / step)
            sys.stdout.write(cl+'\n%12.2f  %7.2f%%  %7.2f%%  %s  %s' % \
            (gbestE, 100.0 * acceptance, 100.0 * improvement,\
            time_string(elapsed), time_string(remain))),
            sys.stdout.flush()
            # with open(self.fdir+'log.csv','a') as f:
            #     f.write('\n'+','.join([str(gbestE),str(acceptance),str(improvement),time_string(elapsed), time_string(remain)]))


    def DPSO(self):

        self.start = time.time()
        step, trials, accepts, improves = 0, 0, 0, 0
        length = self.nNodes - 1
        fig, ax = plt.subplots()
        ax.grid()

        # Create random solutions for each particle
        Swarm = []  # a list of State objects
        for i in range(self.size_population):
            state_idx = self.get_random_solution()
            tempVel = [0] * (length + 1)
            state = Particle(state_idx, tempVel, self.energy(self.getState(state_idx)))

            Swarm.append(state)

        # initialize gbest and pbest
        Swarm.sort()
        gbest = Swarm[0].deepcopy()  # best solution (MIN)
        pbest = Swarm
        self.update(step, gbest.energy, None, None)

        delta_w = (self.w - 0.4) / (self.maxIterations - 1)

        # for each time step (iteration)
        for t in range(self.maxIterations):
            step += 1
            # for each particle in the swarm
            for i in range(self.size_population):
                trials += 1

                previous_route = np.array(Swarm[i].state_idx)
                inertia = self.w * np.array(Swarm[i].velocity)
                personal = self.c1 * np.random.uniform(0, 1) * (np.array(pbest[i].state_idx) - previous_route)
                social = self.c2 * np.random.uniform(0, 1) * (np.array(gbest.state_idx) - previous_route)

                new_velocity = inertia + personal + social

                temp_route = np.add(previous_route, new_velocity)
                _, _, new_route = np.unique(temp_route, return_index=True, return_inverse=True, return_counts=False)

                Swarm[i].velocity = new_velocity.tolist()
                Swarm[i].state_idx = new_route.tolist()

                # gets cost of the current solution
                Swarm[i].update_energy(self.energy(self.getState(Swarm[i].state_idx)))
                cost_current_solution = Swarm[i].energy

                # checks if current solution is pbest solution
                pbest[i].update_energy(self.energy(self.getState(pbest[i].state_idx)))
                if cost_current_solution < pbest[i].energy:
                    pbest[i] = Swarm[i].deepcopy()  # copy of the pbest solution
                    accepts += 1
                self.update(step, pbest[i].energy, accepts / trials, improves / trials, False)
                ax.scatter(t, Swarm[i].energy)

                pbest.sort()
                if pbest[0].energy < gbest.energy:
                    gbest = pbest[0].deepcopy()
                    gbest.velocity = pbest[0].velocity.copy()
                    gbest.update_energy(self.energy(self.getState(gbest.state_idx)))
                    improves += 1
                self.update(step, gbest.energy, accepts / trials, improves / trials, True)
                self.w = self.w - delta_w
            # trials, accepts, improves = 0, 1, 1

        ax.set(xlabel='Iteration', ylabel='Cost',  title='PSO for TSP')
        plt.show()

        self.gbest = gbest.deepcopy()
        if self.save_state_on_exit:
            self.save_state()

        return self.getState(gbest.state_idx), gbest.energy


    def DPSO_Swap(self):
        self.start = time.time()
        step, trials, accepts, improves = 0, 0, 0, 0

        #length = len(self.distance_matrix) - 1
        length = self.nNodes - 1

        # Create random solutions for each particle
        Swarm = []  # a list of State objects
        for i in range(self.size_population):
            tempVel = []
            state_idx = self.get_random_solution()
            for N in range(random.randint(0, length)):  # random initial velocities
                tempVel.append((random.randint(0, length-1), random.randint(0, length-1), self.w))
            state = Particle(state_idx, tempVel, self.energy(self.getState(state_idx)))
            Swarm.append(state)

        # initialize gbest and pbest
        Swarm.sort()
        gbest = Swarm[0].deepcopy()  # best solution (MIN)
        pbest = Swarm
        self.update(step, gbest.energy, None, None)

        # RUNNING THE ALGORITHM:
        # for each time step (iteration)
        for t in range(self.maxIterations):
            step += 1
            # for each particle in the swarm
            # each particle is an object of State
            for i in range(self.size_population):
                trials += 1
                temp_velocity = Swarm[i].velocity
                print("\n Swarm[i].state_idx = ", Swarm[i].state_idx)
                # generates all swap operators to calculate (pbest - x(t-1))
                for n in range(length):
                    if Swarm[i].state_idx[n] != pbest[i].state_idx[n]:
                        # generates swap operator
                        swap_operator = (n, pbest[i].state_idx.index(Swarm[i].state_idx[n]), self.c1)
                        # append swap operator in the list of velocity
                        temp_velocity.append(swap_operator)

                # generates all swap operators to calculate (gbest - x(t-1))
                for n in range(length):
                    if Swarm[i].state_idx[n] != gbest.state_idx[n]:
                        # generates swap operator
                        swap_operator = (n, gbest.state_idx.index(Swarm[i].state_idx[n]), self.c2)
                        # append swap operator in the list of velocity
                        temp_velocity.append(swap_operator)

                # updates velocity: V(t) = w.V(t-1) + (pbest - x(t-1)) + (gbest - x(t-1))
                Swarm[i].velocity = temp_velocity

                # generates new solution for particle: X(t) = X(t-1) + V(t)
                for swap_operator in temp_velocity:
                    if random.random() <= swap_operator[2]:
                        # makes the swap (i <-> j)
                        aux = Swarm[i].state_idx[swap_operator[0]]
                        Swarm[i].state_idx[swap_operator[0]] = Swarm[i].state_idx[swap_operator[1]]
                        Swarm[i].state_idx[swap_operator[1]] = aux

                # gets cost of the current solution
                Swarm[i].update_energy(self.energy(self.getState(Swarm[i].state_idx)))
                cost_current_solution = Swarm[i].energy

                # checks if current solution is pbest solution
                pbest[i].update_energy(self.energy(self.getState(pbest[i].state_idx)))
                if cost_current_solution < pbest[i].energy:
                    pbest[i] = Swarm[i].deepcopy()  # copy of the pbest solution
                    accepts += 1
                self.update(step, pbest[i].energy, accepts / trials, improves / trials, False)


            for i in range(self.size_population):
                gbest.update_energy(self.energy(self.getState(gbest.state_idx)))
                pbest[i].update_energy(self.energy(self.getState(pbest[i].state_idx)))
                if pbest[i].energy < gbest.energy:
                    gbest = pbest[i].deepcopy()  # copy of the pbest solution
                    gbest.update_energy(self.energy(self.getState(gbest.state_idx)))
                    improves += 1
                self.update(step, gbest.energy, accepts / trials, improves / trials, True)

            trials, accepts, improves = 0, 1, 1

        self.gbest = gbest.deepcopy()
        if self.save_state_on_exit:
            self.save_state()


        return self.getState(gbest.state_idx), gbest.energy
