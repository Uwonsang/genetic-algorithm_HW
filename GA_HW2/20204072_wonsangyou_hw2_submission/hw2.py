import matplotlib.pyplot as plt
import random
import utils
from tqdm import tqdm
import glob
import os
import multiprocessing
import time

##parameter
crossover_prob = 0.9
mutation_prob = 0.01
portion = 0.95

def generate_example(data, figure, tournament):
    """Generate example outputs"""
    # generate fitness score trace with some random data
    d1 = list()
    d2 = list()

    # city distance
    distance = utils.read_data(data)
    ######################
    #### RW_selection ####
    ######################
    pop1 = utils.initialize()

    # 6000번 generation
    for _ in tqdm(range(6000)):

        pop1_fitness_list = []
        for ind1 in pop1:
            pop1_fitness_list.append(utils.fitness_function(ind1, distance).cal_fitness())

        tmp_pop1 = utils.tournament_selection(pop1, pop1_fitness_list)
        crossover_pop1 = utils.crossover(tmp_pop1, crossover_prob)
        mutation_pop1 = utils.mutation(crossover_pop1, mutation_prob)
        elisism_pop1 = utils.elisism(pop1, mutation_pop1, distance, portion)
        fitness_avg = utils.get_average(elisism_pop1, distance)
        fitness_best = utils.get_best(elisism_pop1,distance)
        d1.append(fitness_avg)
        d2.append(fitness_best)
        # print('fitness_avg:', fitness_avg)
        pop1 = elisism_pop1


    txt = ""

    for ind1 in elisism_pop1:
        fit1 = utils.fitness_function(ind1, distance).cal_fitness()
        txt += "{},{:.6f}\n".format(ind1, fit1)


    with open(tournament, "w") as f:
        f.write(txt)

    plt.title("Traveling Saleman Problem fitness trace")
    plt.plot(range(6000), d1, label= "Log(Fitness), average")
    plt.plot(range(6000), d2, label= "Log(Fitness), best")
    plt.legend()
    plt.savefig(figure)
    plt.show()

# def train(file_path):
#     total_path = glob.glob("../data(TSP)/*.txt")
#     sorted_total = sorted(total_path)
#     for i in sorted_total:
#         dir, file_name = os.path.split(i)[0], os.path.split(i)[1][5]
#         save_txt = 'fitness-' + file_name + '.txt'
#         save_png = 'fitness-' + file_name + '.png'
#         generate_example(i, save_png, save_txt)
#         print("Done!")


def train(file):
    dir, file_name = os.path.split(file)[0], os.path.split(file)[1].split('.')[0].split('-')[1]
    save_txt = 'fitness-' + file_name + '.txt'
    save_png = 'fitness-' + file_name + '.png'
    print('file_name :', save_txt)
    generate_example(file, save_png, save_txt)
    print("Done!")


if __name__ == '__main__':
    file_path = glob.glob("../data(TSP)/*.txt")
    sorted_file_path = sorted(file_path)
    ## multi_propressing
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map(train, sorted_file_path)
    pool.close()
    pool.join()

