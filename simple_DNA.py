import numpy as np
import matplotlib.pyplot as plt
import os

# constants
N_GENS = 50
DNA_SIZE = 1
DNA_BOUNDS = [0, 5]     # upper and lower bound for DNA values
MU = 100                # population size
LAMBDA = 50             # number of offsprings
IMG_SAVE = "figs/size_1_DNA/"

if not os.path.exists(IMG_SAVE):
    os.mkdir(IMG_SAVE)


def F(x):
    return x * (np.sin(20*x) + np.cos(20*x))

def init_population():
    population = {
        'DNA' : DNA_BOUNDS[1] * np.random.rand(MU, DNA_SIZE),
        'S'   : np.random.rand(MU, DNA_SIZE),
    }
    return population

def select_mates():
    p = np.random.randint(0, MU, size=2)
    return p

def recombine(population, p1, p2):
    child_DNA = [0] * DNA_SIZE
    child_S   = [0] * DNA_SIZE
    for i in range(DNA_SIZE):
        if np.random.rand() < 0.5:
            # select from parent 1
            child_DNA[i] = population['DNA'][p1, i]
            child_S[i] = population['S'][p1, i]
        else:
            # select from parent 2
            child_DNA[i] = population['DNA'][p2, i]
            child_S[i] = population['S'][p2, i]

    return np.asarray(child_DNA), np.asarray(child_S)

def mutate_DNA(DNA, S):
    mutation = S * np.random.randn(1, DNA_SIZE)
    DNA = DNA + mutation
    DNA = np.clip(DNA, *DNA_BOUNDS)
    return DNA

def mutate_s(S):
    S = S + (np.random.rand(1, DNA_SIZE) - 0.5)
    S = np.maximum(S, 0)
    return S

def select_fittest(parents, children):
    population = dict()
    population['DNA'] = np.vstack((parents['DNA'], children['DNA']))
    population['S'] = np.vstack((parents['S'], children['S']))

    fitness = F(population['DNA']).flatten()
    increasing_fitness_index = np.argsort(fitness)
    best_people = increasing_fitness_index[-MU:]

    population['DNA'] = population['DNA'][best_people]
    population['S'] = population['S'][best_people]
    return population


def produce_offsprings(population):
    offsprings = {
        'DNA' : [],
        'S'   : [],
    }
    for i in range(LAMBDA):
        p1, p2 = select_mates()
        child_DNA, child_S = recombine(population, p1, p2)

        child_S = mutate_s(child_S)
        child_DNA = mutate_DNA(child_DNA, child_S)

        offsprings['DNA'].extend(child_DNA)
        offsprings['S'].extend(child_S)

    return offsprings


def main():
    plt.ion()
    x = np.linspace(*DNA_BOUNDS, 200)
    plt.plot(x, F(x))
    P = init_population()
    # iterate through generations of evolution
    for i in range(N_GENS):
        print("Generation: [%3d]" %(i + 1))
        sca = plt.scatter(P['DNA'], F(P['DNA']), s=100, lw=0, c='green', alpha=0.5)
        plt.savefig(IMG_SAVE + "generation_" + str(i + 1) + ".png")
        if i != N_GENS - 1:
            sca.remove()
        new_gen = produce_offsprings(P)
        P = select_fittest(P, new_gen)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
