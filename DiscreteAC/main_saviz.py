import math

# from aco import ACO, Graph
# from plot import plot

from DiscreteAC import aco_saviz as aco
from DiscreteAC import plot

def main():
    cities = []
    points = []
    with open('./data/chn31.txt') as f:
        for line in f.readlines():
            city = line.split(' ')
            cities.append(dict(index=int(city[0]), x=int(city[1]), y=int(city[2])))
            points.append((int(city[1]), int(city[2])))
    rank = len(cities)


    aco_obj = aco.ACO(10, 100, 1.0, 10.0, 0.5, 10, 2)
    graph = aco.Graph(cities, rank)
    path, cost = aco_obj.solve(graph)
    print('cost: {}, path: {}'.format(cost, path))
    plot.plot(points, path)

if __name__ == '__main__':
    main()
