from __future__ import print_function
import math
import random
from simanneal import Annealer
from simanneal import IF_Anneal
import copy

# 13 cities
cities = ['New York', 'Los Angeles', 'Chicago', 'Minneapolis', 'Denver', 'Dallas', 'Seattle', 'Boston',
              'San Francisco', 'St. Louis', 'Houston', 'Phoenix', 'Salt Lake City']
city_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Index of start location
home_idx = 10  # Houston

# Distances in miles between cities, same indexes (i, j) as in the cities array
distance_matrix = \
          [[0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
          [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
          [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
          [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
          [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
          [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
          [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
          [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
          [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
          [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
          [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
          [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
          [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0]]

# create a distance matrix
dist_dict = {}
for i in range(len(cities)):
    dist_dict[cities[i]] = {}
    for j in range(len(cities)):
        dist_dict[cities[i]][cities[j]] = distance_matrix[i][j]


####################################

init_state = copy.deepcopy(cities)
random.shuffle(init_state)

print('Shuffled initial state names = ', init_state)
print('All city names               = ', cities)

# create a new distance matrix
distance_matrix2 = []
AllCities = list(dist_dict.keys())  # names of the cities
new_city_indices = list(range(len(AllCities)))
print('New city indices = ', new_city_indices)

dict_idx = {}
for i in range(len(AllCities)):
    dict_idx[AllCities[i]] = new_city_indices[i]

print('New dict for city indices = ', dict_idx)

print(init_state[2])
print(dict_idx[init_state[2]])

init_state_indices = copy.deepcopy(new_city_indices)
for i in range(len(init_state)):
    init_state_indices[i] = dict_idx[init_state[i]]

print('City indices for shuffled state = ', init_state_indices)

# get city names from indices

new_cities = []
temp = list(dict_idx.keys())
for i in init_state_indices:
    new_cities.append(temp[i])
print('City names of the shuffled city indices = ', new_cities)

###################


for i in range(len(dist_dict)):
    tempRow = []
    for j in range(len(dist_dict)):
        tempRow.append(dist_dict[AllCities[i]][AllCities[j]])
    distance_matrix2.append(tempRow)

print('Newly created distance matrix = ', distance_matrix2)


def get_random_solution(city_indices, home_idx, dict_idx, dist_dict, n_random: int, use_weights: bool = False):

    citiesIdx = city_indices.copy()
    citiesIdx.pop(home_idx)
    random.shuffle(citiesIdx)
    citiesIdx.insert(0, home_idx)
    E = energy(getStateSTR(citiesIdx, dict_idx=dict_idx), dist_dict = dist_dict)
    rand_state_idx = citiesIdx.copy()

    for i in range(n_random-1):
        citiesIdx = city_indices.copy()
        citiesIdx.pop(home_idx)
        if (use_weights == True):
            pass  # state = get_random_solution_with_weights(self.distance_matrix, self.home_idx)
        else:
            # Shuffle cities at random
            random.shuffle(citiesIdx)
            citiesIdx.insert(0, home_idx)
            E_new = energy(getStateSTR(citiesIdx, dict_idx=dict_idx), dist_dict = dist_dict)
            if (E_new < E):
                E = copy.deepcopy(E_new)
                print('Energy = ', E)
                rand_state_idx = citiesIdx.copy()
    # Return the best solution ( Energy)
    return rand_state_idx

def energy(state, dist_dict):
    """Calculates the length of the route."""
    e = 0
    for i in range(len(state)):
        e += dist_dict[state[i-1]][state[i]]
    return e

def getStateSTR(state_idx, dict_idx):
    # get city names from indices
    temp = list(dict_idx.keys())
    state_str = []
    for i in state_idx:
        state_str.append(temp[i])
    return state_str

def getStateIDX(state_str, node_indices, dict_idx):

    state_idx = copy.deepcopy(node_indices)
    for i in range(len(state_str)):
        state_idx[i] = dict_idx[state_str[i]]

    return state_idx

random_solution = get_random_solution(new_city_indices, home_idx = home_idx, dict_idx = dict_idx , dist_dict= dist_dict, n_random = 100)
print('random_solution = ', random_solution)

random_solution_str = getStateSTR(random_solution, dict_idx)
print('random_solution_str = ', random_solution_str)

random_solution_idx = getStateIDX(random_solution_str, city_indices ,dict_idx)
print('random_solution_idx = ', random_solution_idx)