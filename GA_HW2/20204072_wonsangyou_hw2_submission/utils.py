import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import math
from collections import deque


def read_data(filename):
    """read_distance"""
    with open(filename, "r") as f:
        distance_list = []
        for line in f:
            distance_list.append(line)
        distance_list = distance_list[4:]

        distance_table = []
        for distance in distance_list:
            distance_table.append(distance.split(','))

    return distance_table


def initialize():
    """Initialize 100 individuals, each of which consists of city_num : 200"""
    population_num = 100
    city_num = 200

    population = []
    for _ in range(1, population_num+1):
        individual = random.sample(set(np.arange(0, city_num, dtype=int)), city_num) #route generation
        population.append(individual) #make population

    return np.array(population)


class fitness_function:
    def __init__(self, route, distance_table):
        self.route = route
        self.distance_table = distance_table
        self.distance = 0.0
        self.fitness= 0.0

    def cal_fitness(self):
        """Calculate fitness value of an individual."""
        if self.distance == 0.0:
            path_distance = 0.0
            for i in range(1, len(self.route)):
                k = int(self.route[i - 1])
                l = int(self.route[i])
                path_distance += float(self.distance_table[k][l])

            self.distance = path_distance

        return math.log(self.distance, 2)


def get_average(population, distance):
    avg = []
    for ind1 in population:
        fitness = fitness_function(ind1, distance).cal_fitness()
        avg.append(fitness)

    return sum(avg)/len(avg)


def get_best(population, distance):
    best = []
    for ind1 in population:
        fitness = fitness_function(ind1, distance).cal_fitness()
        best.append(fitness)

    return np.min(best)


def RW_selection(file, fitness_list):
    file = copy.deepcopy(file)
    data_len = len(file)

    fitness_list = [(float(i)-min(fitness_list))/ (max(fitness_list) - min(fitness_list)) for i in fitness_list]

    tmp_population = []
    for i in range(data_len):
        x = np.random.rand()
        k = 0
        total_fitness = sum(fitness_list)

        while k < data_len and x > sum(fitness_list[:k+1]) / total_fitness:
            k = k+1
        tmp_population.append(file[k])

    return tmp_population


def tournament_selection(file, fitness_list):
    file = copy.deepcopy(file)
    data_len = len(file)

    tmp_population = []

    for i in range(data_len):
        x = random.randrange(0, data_len)

        if fitness_list[i] < fitness_list[x]:
            tmp_population.append(file[i])
        else:
            tmp_population.append(file[x])

    return tmp_population


def crossover(file, crossover_prob):
    file = copy.deepcopy(file)

    data_len = len(file)
    gene_len = len(file[0])  ##200

    for i in range(data_len):

        if np.random.rand() < crossover_prob:
            exclude = [i]
            chromosome_1 = file[i]
            chromosome_2 = file[random.choice([i for i in range(100) if i not in exclude])]

            # choose random point
            start, end = sorted(np.random.choice(gene_len, 2))

            inherit_gene_1 = []
            offspring = np.zeros(gene_len)
            for j in range(start, end):
                offspring[j] = chromosome_1[j]
                inherit_gene_1.append(chromosome_1[j])

            inherit_gene_2 = np.setdiff1d(chromosome_2, inherit_gene_1)
            inherit_gene_2_index = sorted([np.where(chromosome_2==i)[0][0] for i in inherit_gene_2])
            inherit_gene_final = chromosome_2[inherit_gene_2_index]


            offspring[:start] = inherit_gene_final[:start]
            offspring[end:gene_len] = inherit_gene_final[start:gene_len]

            file[i] = offspring

    return file

def mutation(file ,mutation_prob):

    offspring = copy.deepcopy(file)
    data_len = len(offspring)
    gene_len = len(offspring[0])

    for i in range(data_len):
        if np.random.rand() <= mutation_prob:
            random_prob_array = np.random.randint(0, gene_len, size=2)
            tmp = offspring[i][random_prob_array[0]]
            offspring[i][random_prob_array[0]] = offspring[i][random_prob_array[1]]
            offspring[i][random_prob_array[1]] = tmp

    return offspring


def elisism(pop, file, distance, portion):
    pop = copy.deepcopy(np.array(pop))
    file = copy.deepcopy(np.array(file))
    data_len = len(file)


    population = {}
    for i in range(data_len):
        population[i] = fitness_function(pop[i], distance).cal_fitness()

    population = sorted(population.items(),key = lambda item: item[1])
    sorted_population = population[:int(data_len * portion)]
    sorted_index = [index for index, value in sorted_population]

    selected_population = pop[sorted_index]
    random_index = random.sample(range(data_len), data_len - int(data_len * portion))
    left_population = file[random_index]

    offspring = np.concatenate((selected_population, left_population), axis=0)

    return offspring




