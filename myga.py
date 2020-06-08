"""
.. module:: myga.py
    :synopsis: Module implements a general Genetic Algorithm.
    It implements basic genetic operators and some selection
    functions.
.. moduleauthor:: Rubén Ibarrondo (rubenibarrondo@gmail.com)
"""

import numpy as np
import sys

# ============================================
#
#                GA operators
#
# ============================================


def crossover(c1, c2, i):
    """
        Returnes the crossovers at i of chromosomes
        c1 and c2.
    Args:
        c1 (list): chromosome representation of member 1.
        c2 (list): chromosome representation of member 2.
        i (int): index where crossover will be performed.
    Returns:
        c12 (list): array chromosome representation of the
            offspring between 1 and 2, with 1 first.
            c12 = c1[:i]+c2[i:]
        c21 (list): array chromosome representation of the
            offspring between 1 and 2, with 2 first.
            c21 = c2[:i]+c1[i:]
    Rises:

    """
    if len(c1) != len(c2):
        raise Exception("c1 and c2 must be same lenght")
    return c1[:i]+c2[i:], c2[:i]+c1[i:]


def mutation(c, i):
    """
        Mutates the chromosome c at position i.
    Args:
        c (list): chromosome representation of a member.
        i (int): index where mutation will be performed.
    Returns:
        mutc (list): chromosome representation of the
            mutated member.
    Rises:

    """
    mutc = c[:]
    mutc[i] = c[i] ^ 1
    return mutc


def fitnessprop_selection(fitness):
    """
        Implements fitness proportional selection. It only works with
        positive fitness values.
    Args:
        fitness (array): The fitness of each member
    Returns:
        selcp (list): The selection probability for each member
    Rises:
        NegativeFitnessError: Proportional selection only works with
            positive fitness.

    """
    if min(fitness) < 0:
        sys.exit("NegativeFitnessError encountered.")
    fsum = sum(fitness)
    return [f/fsum for f in fitness]


def difbased_selection(fitness):
    """
        Implements fitness difference based selection.
    Args:
        fitness (array): The fitness of each member
    Returns:
        selcp (list): The selection probability for each member
    Rises:

    """
    fmin = min(fitness)
    if fmin == max(fitness):
        return [1/len(fitness) for f in fitness]
    fdifs = [(f-fmin) for f in fitness]
    fsum = sum(fdifs)
    return [f/fsum for f in fdifs]


# ============================================
#
#          General Genetic Algorithm
#
# ============================================


def GeneticOptimization(fitness_function, args=(),
                        chromosome_length=32,
                        populationNumber=100, generationNumber=100,
                        pc=0.7, pm=0.001,
                        selection_probability=difbased_selection,
                        getEvolution=0):
    """
        This function implements a general Genetic Algorithm. It uses fitness
        idependent mutation and crossover probabilities.

    Args:
        fitness_function (function): The function that computes the fitness for
            each member. It takes an entire population, as a list of
            chromosomes, and output their fitness in the same order as a list
            of floats. If needed extra arguments may be added using args.
        args (tuple, optional): Contains the extra arguments for the
            fitness_function.
            Defaults to (), empty tuple.
        chromosome_length (int, optional): The length for the chromosome of the
            members.
            Defaults to 32.
        populationNumber (int, optional): The number of members in the evolving
            population.
            Defaults to 100.
        generationNumber (int, optional): The number of generations the
            algorithm will evolve.
            Defaults to 100.
        pc (float, optional): crossover probability, float number in range
            [0.0, 1.0).
            Defaults to  0.7
        pm (float, optional): mutation probability, float number in range
            [0.0,1.0).
            Defaults to 0.001
        selection_probability (function, optional): The function that computes
            the selection probability. It takes the fitness distribution as a
            list and returns a list with values in range [0.0, 1.0) which add
            up tu 1 as a probability distribution. Defaults to
            difbased_selection.
        getEvolution (int, optional): Allows to get the evolution of the
            population. It's value defines the frequency for making records.
            Defaults to 0, no record.

    Returns:
        final_pop (list): The final population as a list for chromosomes.
        poprecord (list, optional): The record of saved populations during the
            evolution.
            If getEvolution is 0 it is not computed nor returned.
            Its length is 1+(generationNumber-1)//getEvolution, the first
            record is done with the initial population. The final population is
            never saved here.

    Rises:

    """

    if getEvolution != 0:
        poprecord = []

    # 1. Initial population
    population = []
    for c in range(populationNumber):
        chromosome = list(np.random.randint(0, 2, chromosome_length))
        population.append(chromosome)

    for g in range(generationNumber):
        if getEvolution != 0 and g % getEvolution == 0:
            poprecord.append(population)

        # 2. Calculate fitness function
        fitness = fitness_function(population, *args)

        # 3. Offspring creation
        offspring = []
        while len(offspring) < populationNumber:

            # 3.a Parent chromosome selection
            i, j = np.random.choice(range(len(population)),
                                    p=selection_probability(fitness),
                                    size=2)
            ci, cj = population[i], population[j]

            # 3.b Apply crossover or not
            rc = np.random.random()
            if rc < pc:
                index = np.random.randint(len(ci))
                newci, newcj = crossover(ci, cj, index)
            else:
                newci, newcj = ci[:], cj[:]

            # 3.c Apply mutation or not
            for index in range(len(cj)):
                rm = np.random.random()
                if rm < pm:
                    newci = mutation(newci, index)
                    newcj = mutation(newcj, index)

            offspring.append(newci)
            offspring.append(newcj)

            # This would be used when populaitonNumber is odd
            while len(offspring) > populationNumber:
                index = np.random.randint(len(offspring))
                offspring.pop(index)
        population = offspring

    if getEvolution != 0:
        return population, poprecord
    else:
        return population
