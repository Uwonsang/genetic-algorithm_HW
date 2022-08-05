import matplotlib.pyplot as plt
import random
import numpy as np
import copy

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

        if fitness_list[i] > fitness_list[x]:
            tmp_population.append(file[i])
        else:
            tmp_population.append(file[x])

    return tmp_population


def crossover(file, crossover_prob):
    file = copy.deepcopy(file)

    data_median = len(file) //2 ##50
    gene_len = len(file[0]) ##10000

    for i in range(data_median):

        if np.random.rand() < crossover_prob:
            ######cross_point3 적용하기#####

            part1 = file[i]
            part2 = file[i + data_median]

            individual_1 = list(part1)
            individual_2 = list(part2)

            gene_position = np.random.choice(gene_len, 3)
            position_list = sorted(gene_position)

            for k in range(position_list[0], position_list[1]):  # split_position 이후의 데이터 쌓기
                tmp = individual_1[k]
                individual_1[k] = individual_2[k]
                individual_2[k] = tmp

            for j in range(position_list[2], gene_len):
                tmp = individual_1[j]
                individual_1[j] = individual_2[j]
                individual_2[j] = tmp

            file[i] = ''.join(individual_1)
            file[i + data_median] = ''.join(individual_2)

    return file

def mutation(file ,mutation_prob):
    offspring = copy.deepcopy(file)
    data_len = len(offspring)
    gene_len = len(offspring[0])

    for i in range(data_len):
        for k in range(gene_len):
            if np.random.rand() <= mutation_prob:
                if offspring[i][k] == '0':
                    offspring[i] = offspring[i][:k] + '1' + offspring[i][k+1:]
                elif offspring[i][k] == '1':
                    offspring[i] = offspring[i][:k] + '0' + offspring[i][k+1:]

    return offspring


def get_average(population, spec):
    avg = []
    for ind1 in population:
        fitness = fitness_function(ind1, *spec)
        avg.append(fitness)

    return sum(avg)/len(avg)


def read_data(filename):
    """Parse problem specifications from the data file."""
    with open(filename, "r") as f:
        # header
        for line in f:
            iwp = line.strip().split()
            if len(iwp) >= 4 and iwp[2] == "capacity":
                capacity = float(iwp[3])
            elif iwp == ["item_index", "weight", "profit"]:
                table = True
                break
        if not table:
            raise ValueError("table not found.")
        # body
        weights = []
        profits = []
        for line in f:
            i, w, p = line.strip().split()
            weights.append(float(w))
            profits.append(float(p))
    return capacity, weights, profits

def fitness_function(individual, capacity, weights, profits):
    """Calculate fitness value of an individual."""
    sum_weight = 0
    sum_profit = 0
    for bit, weight, profit in zip(individual, weights, profits):
        if bit == "1":
            sum_weight += weight
            sum_profit += profit

    fitness = sum_profit if sum_weight <= capacity else 0
    return fitness

def initialize():
    """Initialize 100 individuals, each of which consists of 10000 bits"""
    population = []
    for _ in range(100):
        individual = ""
        for _ in range(10000):
            individual += "1" if random.random() < 0.5 else "0"
        population.append(individual)
    return population