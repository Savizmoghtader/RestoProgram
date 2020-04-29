import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt

# class City:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#     def distance(self, city2):
#         xDis = abs(self.x - city2.x)
#         yDis = abs(self.y - city2.y)
#         distance = np.sqrt((xDis ** 2) + (yDis ** 2))
#         return distance
#
#     def __repr__(self):
#         return "(" + str(self.x) + "," + str(self.y) + ")"

# class Fitness:
#     def __init__(self, route):
#         self.route = route
#         self.fitness = 0.0
#         self.energy = 0
#
#     def routeDistance(self):
#         if self.energy == 0:
#             pathDistance = 0
#             for i in range(0, len(self.route)):
#                 fromCity = self.route[i]
#                 toCity = None
#                 if i + 1 < len(self.route):
#                     toCity = self.route[i + 1]
#                 else:
#                     toCity = self.route[0]
#                 pathDistance += fromCity.distance(toCity)
#             self.distance = pathDistance
#
#         return self.energy
#
#     def routeFitness(self):
#         if self.fitness == 0:
#             self.fitness = 1 / float(self.routeDistance())
#         return self.fitness

def createRoute(state):
    route = random.sample(state, len(state)) #E.g. list1 = [1, 2, 3, 4, 5] /print(sample(list1,3)) = [2, 3, 5]
    return route


def initialPopulation(popSize, state):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(state))
    return population


def rankRoutes(population):
    fitnessResults = {}
    energy = [10541, 10442, 75634754, 10644, 135364, 105634, 2434245, 105348, 156409, 1054400, 1054541, 1012442, 7356754, 1062444, 13564, 1056, 243445, 10258, 154609, 112000]
    for i in range(0,len(population)):
        fitnessResults[i] = energy[i]
    popRanked = sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = False)
    return popRanked # sort a list of tuples on the second key.  The first key is the route

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    # print(df)

    for i in range(0, eliteSize):
       selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
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


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(state, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, state)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

    return bestRoute

###################################################################################################
cityList = []

# for i in range(0,25):
#     cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

# init_state = [(((759564.7, 193863.6), (759486.2, 193874.4)), 0), (((756566.7, 190985.3), (756566.83, 191085.62)), 2), (((750892.8, 188259.0), (750846.7, 188266.5)), 1), (((750758.67, 185809.8), (750687.71, 185933.99)), 2), (((750173.5, 187788.7), (750254.0, 187818.8)), 1), (((761299.9, 197672.1), (760157.04, 195601.1)), 2), (((760146.99, 195565.56), (760061.78, 195228.28)), 2), (((750205.1, 187513.1), (750274.52, 187591.83)), 2), (((760072.86, 195139.91), (760316.99, 195480.22)), 2), (((759276.93, 193205.89), (758864.56, 192928.38)), 2), (((759486.67, 194199.2), (759684.8, 194851.49)), 2), (((750205.1, 187513.1), (750032.6, 187537.7)), 2), (((757295.4, 191208.2), (756999.6, 191494.5)), 2), (((759199.86, 192428.21), (759496.12, 193153.65)), 2), (((750274.52, 187591.83), (750970.7, 188290.6)), 2), (((750029.5, 187527.68), (750111.3, 187316.8)), 2), (((750183.5, 187097.9), (750024.54, 187597.17)), 2), (((757048.2, 190761.8), (756692.3, 191010.4)), 2), (((750418.28, 187694.49), (750372.0, 187763.7)), 2), (((759684.8, 194851.49), (759921.1, 195903.9)), 2), (((750846.7, 188266.5), (750720.3, 188285.3)), 1), (((757051.9, 190757.8), (757384.9, 191117.0)), 2), (((757295.4, 191208.2), (757357.01, 191276.99)), 2), (((756263.0, 190580.5), (756566.7, 190985.3)), 2), (((760653.9, 196792.7), (761122.2, 197363.1)), 2), (((756566.83, 191085.62), (757125.6, 191859.3)), 2), (((757191.29, 190611.49), (757051.9, 190757.8)), 1), (((760395.1, 196965.5), (760902.2, 197368.6)), 2), (((758869.98, 192919.98), (759283.67, 193198.51)), 2)]
init_state = [(((11, 12), (13, 14)), 0), (((21, 22), (23, 24)), 1), (((31, 32), (33, 34)), 2), (((41, 42), (43, 44)), 3), (((51, 52), (53, 54)), 4), (((61, 62), (63, 64)), 5), (((71, 72), (73, 74)), 6), (((81, 82), (83, 84)), 7), (((91, 92), (93, 94)), 8), (((101, 102), (103, 104)), 9), (((111, 112), (113, 114)), 10)]

geneticAlgorithm(state=init_state, popSize=20, eliteSize=2, mutationRate=0.01, generations=10)