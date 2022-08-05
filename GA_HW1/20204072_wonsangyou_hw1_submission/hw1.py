import matplotlib.pyplot as plt
import random
import utils
from tqdm import tqdm

##parameter
crossover_prob = 0.9
mutation_prob = 0.01


def generate_example(data, figure, tournament, roulette):
    """Generate example outputs"""
    # generate fitness score trace with some random data
    # d1 = [250000 + i * 100 + random.randrange(-800, 800) for i in range(100)]
    # d2 = [260000 - (i-100) ** 2 + random.randrange(-800, 800) for i in range(100)]
    d1 = list()
    d2 = list()

    # export two random populations as a txt file
    spec = utils.read_data(data)

    ######################
    #### RW_selection ####
    ######################
    pop1 = utils.initialize()

    # for 문으로 100 반복
    for _ in tqdm(range(100)):

        pop1_fitness_list = []
        for ind1 in pop1:
            pop1_fitness_list.append(utils.fitness_function(ind1, *spec))

        tmp_pop1 = utils.RW_selection(pop1, pop1_fitness_list)
        crossover_pop1 = utils.crossover(tmp_pop1, crossover_prob)
        mutation_pop1 = utils.mutation(crossover_pop1, mutation_prob)
        fitness_avg = utils.get_average(mutation_pop1, spec)
        d1.append(fitness_avg)

        pop1 = mutation_pop1


    txt = ""

    for ind1 in mutation_pop1:
        fit1 = utils.fitness_function(ind1, *spec)
        txt += "{},{:.6f}\n".format(ind1, fit1)


    with open(roulette, "w") as f:
        f.write(txt)



    ##############################
    #### tournament_selection ####
    ##############################

    pop2 = utils.initialize()

    # for 문으로 100 반복
    for _ in tqdm(range(100)):

        pop2_fitness_list = []
        for ind2 in pop2:
            pop2_fitness_list.append(utils.fitness_function(ind2, *spec))

        tmp_pop2 = utils.tournament_selection(pop2, pop2_fitness_list)
        crossover_pop2 = utils.crossover(tmp_pop2, crossover_prob)
        mutation_pop2 = utils.mutation(crossover_pop2, mutation_prob)
        fitness_avg = utils.get_average(mutation_pop2, spec)
        d2.append(fitness_avg)

        pop2 = mutation_pop2

    txt = ""
    for ind2 in mutation_pop2:
        fit2 = utils.fitness_function(ind2, *spec)
        txt += "{},{:.6f}\n".format(ind2, fit2)

    with open(tournament, "w") as f:
        f.write(txt)


    plt.title("0/1 Knapsack fitness value trace")
    plt.plot(range(100), d1, label="Roulette Wheel Selection")
    plt.plot(range(100), d2, label="Pairwise Tournament Selection")
    plt.legend()
    plt.savefig(figure)
    plt.show()

if __name__ == '__main__':
    generate_example("../Data(0-1Knapsack).txt", "trace.png", "tournament.txt", "roulette.txt")
    print("Done!")