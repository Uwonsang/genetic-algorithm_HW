import numpy as np
import copy
import random
from scipy.spatial.distance import hamming
import math
import matplotlib.pyplot as plt



def binarylist_to_int(value):
    return int("".join(str(x) for x in value), 2)


def formula(individual, x):
    accum_index = 0
    alpha = individual[accum_index:accum_index + 1]
    alpha = -1 if alpha == 0 else 1
    accum_index += 1

    beta = individual[accum_index:accum_index + 5]
    beta = binarylist_to_int(beta)
    accum_index += 5

    theta = individual[accum_index:accum_index + 5]
    theta = binarylist_to_int(theta)
    accum_index += 5

    delta = individual[accum_index:accum_index + 5]
    delta = binarylist_to_int(delta)
    accum_index += 5

    predict_y = alpha * (abs(x) ** (beta)) * math.sin(theta * x + delta)

    return predict_y

def write_formula(individual, x_list, filename):

    for x in x_list:
        accum_index = 0
        alpha = individual[accum_index:accum_index + 1]
        alpha = -1 if alpha == 0 else 1
        accum_index += 1

        beta = individual[accum_index:accum_index + 5]
        beta = binarylist_to_int(beta)
        accum_index += 5

        theta = individual[accum_index:accum_index + 5]
        theta = binarylist_to_int(theta)
        accum_index += 5

        delta = individual[accum_index:accum_index + 5]
        delta = binarylist_to_int(delta)
        accum_index += 5

    name = filename.split('.')[1][10:]

    with open(f'formula_{name}.txt', 'w+') as f:
        f.write(f'Hypothesis: {alpha} X (|x|^{beta}) X sin({theta}x + {delta})')


def regression(individual, target):

    x_list = list()
    y_list = list()
    predict_y_list = list()

    for x, y in target.to_numpy():
        x_list.append(x)
        y_list.append(y)
        predict_y_list.append(formula(individual, x))

    return x_list, y_list, predict_y_list


def initialize(seed_num=42):
    # set seed & initialize
    rng = np.random.default_rng(seed=seed_num)
    pop = rng.integers(2, size=(500, 16))
    # random.seed(seed_num)
    # pop = np.random.randn(10, 4)
    return pop


def fitness_function(individual, target):

    total_loss = 0
    for x, y in target.to_numpy():

        accum_index = 0
        alpha = individual[accum_index:accum_index+1]
        alpha = -1 if alpha == 0 else 1
        accum_index += 1

        beta = individual[accum_index:accum_index+5]
        beta = binarylist_to_int(beta)
        accum_index += 5

        theta = individual[accum_index:accum_index+5]
        theta = binarylist_to_int(theta)
        accum_index += 5

        delta = individual[accum_index:accum_index+5]
        delta = binarylist_to_int(delta)
        accum_index += 5

        predict = alpha * (abs(x) ** (beta)) * math.sin( theta * x + delta)

        loss = abs(y - predict) ** 2
        total_loss += loss

    total_loss /= len(target)

    return total_loss


def zeta(pop, D=30):
    pop = copy.deepcopy(pop)
    pop_length = len(pop[0])

    z_list = []
    for individual1 in pop:
        z = 0
        for individual2 in pop:
            h_distance = int(hamming(individual1, individual2) * pop_length)
            if h_distance < int(D):
                z += (1 - (h_distance)/ int(D))
        z_list.append(z - 1)

    return np.array(z_list)


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

    return np.array(tmp_population)


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

            file[i] = individual_1
            file[i + data_median] = individual_2

    return file


def mutation(file, mutation_prob):
    offspring = copy.deepcopy(file)
    data_len = len(offspring)
    gene_len = len(offspring[0])

    for i in range(data_len):
        for k in range(gene_len):
            if np.random.rand() <= mutation_prob:
                if offspring[i][k] == 0:
                    offspring[i] = np.concatenate((offspring[i][:k], np.array([1]), offspring[i][k+1:]), axis=0)
                elif offspring[i][k] == 1:
                    offspring[i] = np.concatenate((offspring[i][:k],  np.array([0]), offspring[i][k+1:]), axis=0)

    return offspring


def overlap(origin_pop, offspring, fitness_list, portion):
    pop = copy.deepcopy(np.array(origin_pop))
    offspring = copy.deepcopy(np.array(offspring))
    data_len = len(pop)

    population = {}
    for i in range(data_len):
        population[i] = fitness_list[i]

    new_population = sorted(population.items(),key = lambda item: item[1], reverse=False)
    sorted_population = new_population[:int(data_len * portion)]
    sorted_index = [index for index, value in sorted_population]

    selected_population = pop[sorted_index]
    # random_index = random.sample(range(data_len), data_len - int(data_len * portion))
    left_index = np.array(list(set(range(data_len)) - set(sorted_index)))
    left_population = offspring[left_index]

    offspring = np.concatenate((selected_population, left_population), axis=0)

    return offspring

def get_average(population, target):
    avg = []
    for ind1 in population:
        fitness = fitness_function(ind1, target)
        avg.append(fitness)

    return sum(avg)/len(avg)


def get_best(population, target):
    best = []
    for ind1 in population:
        fitness = fitness_function(ind1, target)
        best.append(fitness)

    return np.min(best)
