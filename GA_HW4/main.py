import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils
import pandas as pd

##parameter
num_generation = 500
ratio = 0.95
crossover_prob = 0.9
mutation_prob = 0.01
D = 30



def generation(population, target):

    fitness_list = list()
    for ind in range(len(population)):
        fitness_list.append(utils.fitness_function(population[ind], target))
    fitness_array = np.array(fitness_list)

    zeta_list = utils.zeta(population, D)
    new_fitness_list = fitness_array / zeta_list

    tournament_pop = utils.tournament_selection(population, new_fitness_list)
    crossover_pop = utils.crossover(tournament_pop, crossover_prob)
    mutation_pop = utils.mutation(crossover_pop, mutation_prob)
    final_pop = utils.overlap(population, mutation_pop, new_fitness_list, ratio)

    average = utils.get_average(final_pop, target)
    best = utils.get_best(final_pop, target)

    return final_pop, average, best


def report_result(filename, population, target):

    x_list = list()
    y_list = list()
    predict_y_list = list()
    fitness_list = list()

    for ind in range(len(population)):
        x, y, predict_y = utils.regression(population[ind], target)
        fitness_list.append(utils.fitness_function(population[ind], target))
        x_list.append(x)
        y_list.append(y)
        predict_y_list.append(predict_y)

    best_index = fitness_list.index(np.min(fitness_list))
    plt.scatter(x_list[best_index], y_list[best_index], label='GT', c='blue')
    plt.scatter(x_list[best_index], predict_y_list[best_index], label='predict', c='red')
    plt.show()
    utils.write_formula(population[best_index], x_list[best_index], filename)



def main():
    target_filename = './data(gp)/data-gp2.txt'
    target = pd.read_csv(target_filename, sep=",")

    population = utils.initialize(seed_num=42)

    for _ in tqdm(range(num_generation)):
        offspring_pop, avg, best = generation(population, target)
        population = offspring_pop

    report_result(target_filename, population, target)

if __name__ == "__main__":
    main()
