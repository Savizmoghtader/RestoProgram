from __future__ import print_function
import math
import random
import copy
import IF_DPSO
# from simanneal import IF_Anneal

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

init_state_str = copy.deepcopy(cities)
random.shuffle(init_state_str)

pso = IF_DPSO.DPSO_Optimizer(init_state_str, dist_dict, fdir='savedData')
pso.copy_strategy = "slice"
state, e = pso.DPSO()

while state[0] != cities[home_idx]:
    state = state[1:] + state[:1]  # rotate NYC to start
print()
print("%i mile route with PSO:" % e)
print(" ➞  ".join(state))

# annealer = IF_Anneal.My_Anneal_Interface(init_state_str, dist_dict, fdir='savedData')
# annealer.set_schedule(annealer.auto(minutes=0.2))
# # since our state is just a list, slice is the fastest way to copy
# annealer.copy_strategy = "slice"
# state, e = annealer.anneal()
#
# while state[0] != cities[home_idx]:
#     state = state[1:] + state[:1]  # rotate NYC to start
#
# print()
# print("%i mile route with SA:" % e)
# print(" ➞  ".join(state))