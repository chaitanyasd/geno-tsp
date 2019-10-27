import math
import random
import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt
from settings import *

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        distX = abs(self.x - city.x)
        distY = abs(self.y - city.y)
        distance = np.sqrt(np.square(distX) + np.square(distY))
        return distance

    @property
    def coordinates(self):
        return "({}, {})".format(self.x, self.y)

    def __repr__(self):
        return "({}, {})".format(self.x, self.y)

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
    
    def routeDistance(self):
        totalDistance = 0
        routeLength = len(self.route)

        if routeLength != 0:
            for i in range(routeLength):
                sourceCity = self.route[i]
                destinationCity = None
                if i + 1 < routeLength:
                    destinationCity = self.route[i+1]
                else:
                    destinationCity = self.route[0]
                totalDistance += sourceCity.distance(destinationCity)

        self.distance = totalDistance
        return self.distance
    
    def routefitness(self):
        rDistance = self.routeDistance()
        if rDistance != 0:
            self.fitness = 1 / float(rDistance)
        return self.fitness

def generateRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def createInitialPopulation(populationSize, cityList):
    population = []
    for i in range(populationSize):
        population.append(generateRoute(cityList))
    return population

def rankRoutes(population):
    populationFitness = dict()
    for i in range(len(population)):
        populationFitness[i] = Fitness(population[i]).routefitness()
    return sorted(populationFitness.items(), key=operator.itemgetter(1), reverse=True)

def selectBestRoutes(rankedRoutes, eliteSize, type=0):
    selectionResults = []

    # roulette wheel selection
    if type == 0:
        df = pd.DataFrame(np.array(rankedRoutes), columns=["Index","Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()

        for i in range(eliteSize):
            selectionResults.append(rankedRoutes[i][0])
        for i in range(len(rankedRoutes) - eliteSize):
            pick = random.uniform(min(df['cum_sum']), max(df['cum_sum']))
            for i in range(len(rankedRoutes)):
                if pick <= df.iat[i,2]:
                    selectionResults.append(rankedRoutes[i][0])
                    break
    # tournament selection
    else:
        for i in range(eliteSize):
            selectionResults.append(rankedRoutes[i][0])
        for i in range(len(rankedRoutes) - eliteSize):
            tournmentCandidates = dict()
            k = int(len(rankedRoutes) * 0.25)       # 25% of the total condidates in the route
            for i in range(k):
                index = random.randint(0, len(rankedRoutes) - 1)
                tournmentCandidates[rankedRoutes[index][0]] = rankedRoutes[index][1]
            sortedCandidates = sorted(tournmentCandidates.items(), key=operator.itemgetter(1), reverse=True)
            selectionResults.append(sortedCandidates[0][0])
            
    return selectionResults

def generatMatingPool(population, selectionResults):
    matingPool = []
    for i in range(len(selectionResults)):
        selectionIndex = selectionResults[i]
        matingPool.append(population[selectionIndex])
    return matingPool

def breedIndivisuals(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    startIndex = random.randint(0, len(parent1))
    endIndex = random.randint(0, len(parent1))

    startGene, endGene = min(startIndex, endIndex), max(startIndex, endIndex)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [gene for gene in parent2 if gene not in childP1]
    child = childP1 + childP2
    return child

def breedPopulation(matingPool, eliteSize):
    children = []
    newPopulationSize = len(matingPool) - eliteSize

    for i in range(eliteSize):
        children.append(matingPool[i])
    
    for i in range(newPopulationSize):
        child = breedIndivisuals(matingPool[i], matingPool[len(matingPool)-i-1])
        children.append(child)

    return children

def mutateIndivisual(indivisual, mutationRate):
    for i in range(len(indivisual)):
        if random.random() < mutationRate:
            j = random.randint(0, len(indivisual) - 1)
            indivisual[i], indivisual[j] = indivisual[j], indivisual[i]
    return indivisual

def mutatePopulation(population, mutationRate):
    mutatedPopulation = []
    for i in range(len(population)):
        mutatedIndivisual = mutateIndivisual(population[i], mutationRate)
        mutatedPopulation.append(mutatedIndivisual)
    return mutatedPopulation

def nextGeneration(currentGeneration, eliteSize, mutationRate):
    rankedPopulation = rankRoutes(currentGeneration)
    selectedRoutes = selectBestRoutes(rankedPopulation, eliteSize, selectionMethod)
    matingPool = generatMatingPool(currentGeneration, selectedRoutes)
    children = breedPopulation(matingPool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(cityList, populationSize, eliteSize, mutationRate, numberOfgenerations):
    progress = []
    population = createInitialPopulation(populationSize, cityList)
    print ("Initial distance : {}".format(1 / rankRoutes(population)[0][1]))
    progress.append(1 / rankRoutes(population)[0][1])

    for i in range(numberOfgenerations):
        population = nextGeneration(population, eliteSize, mutationRate)
        print ("Generation {}  Distance : {}".format(i , 1 / rankRoutes(population)[0][1]))
        progress.append(1 / rankRoutes(population)[0][1])
    
    bestRouteInfo = rankRoutes(population)
    print ("Final distance : {}".format(1 / bestRouteInfo[0][1]))
    bestRoute = population[bestRouteInfo[0][0]]
    print ("Best route : {}".format(str(bestRoute)))

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

def generateCityList():
    cityList = []
    for i in range(numberOfCities):    
        city = City(random.randint(0, maxYCoordinate), random.randint(0, maxYCoordinate))
        cityList.append(city)
    return cityList

if __name__ == '__main__':
    random.seed(randomSeedValue)
    cityList = generateCityList()
    geneticAlgorithm(cityList, populationSize, eliteSize, mutationRate, generations)