import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt

#
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
    energy = [101, 102, 756754, 104, 13564, 106, 24345, 108, 109, 6]
    for i in range(0,len(population)):
        fitnessResults[i] = energy[i]
    popRanked = sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = False)
    return popRanked # sort a list of tuples on the second key.  The first key is the route

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum() # todo what is this?

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

init_state = [(((6000.0, 4000.0), (6000.0, 10000.0)), 0), (((6000.0, 10000.0), (12000.0, 18000.0)), 0),
 (((6000.0, 10000.0), (0.0, 18000.0)), 2), (((3000.0, 0.0), (0.0, 18000.0)), 2)]

geneticAlgorithm(state=init_state, popSize=10, eliteSize=2, mutationRate=0.01, generations=20)