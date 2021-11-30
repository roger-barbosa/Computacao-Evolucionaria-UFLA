import numpy as np
from numpy.random import uniform
from math import exp
from math import cos
from math import floor
from math import pi
from math import e
from math import sin
from math import sqrt


def affinity(p_i, prefix):
    """
    Description
    -----------
    Return the affinity of one subject.
    
    Parameters
    -----------
    p_i: numpy.array
        Subject of a population.
    
    Return
    -----------
    return: float
        Affinity of the subject passed as parameter.
    
    """
    if prefix == 'sch_':
        [x, y] = p_i   # Schaffer
        num = (np.sin(np.sqrt((x**2 + y**2)))**2) - 0.5
        den = (1 + 0.001*(x**2 + y**2))**2
        ret = (1 + 0.5 + num/den)
    elif prefix == 'ros_':
        ret = 0
        x = p_i
        for i in range(len(x) - 1):
            ret += 100 * (x[i + 1] - x[i] * x[i]) ** 2 + (x[i] - 1) ** 2
    elif prefix == 'ras_':
        x = p_i
        for item in x:
            assert x <= 5.12 and x >= -5.12, 'input exceeds bounds of [-5.12, 5.12]'
        ret = len(x) * 10.0 + sum([item * item - 10.0 * cos(2.0 * pi * item) for item in x])
    elif prefix == 'ack_':
        [x, y] = p_i   # Schaffer
        ret = (-20 * exp(-0.2 * sqrt(0.5 * (x*x + y*y))) - exp(0.5 * (cos(2.0*pi*x) + cos(2*pi*y))) + e + 20)

    return ret

def create_random_cells(population_size, problem_size, b_lo, b_up):
    population = [uniform(low=b_lo, high=b_up, size=problem_size) for x in range(population_size)]
    
    return population

def clone(p_i, clone_rate):
    clone_num = int(clone_rate / p_i[1])
    clones = [(p_i[0], p_i[1]) for x in range(clone_num)]
    
    return clones

def hypermutate(p_i, mutation_rate, b_lo, b_up, prefix):
    if uniform() <= p_i[1] / (mutation_rate * 100):
        ind_tmp = []
        for gen in p_i[0]:
            if uniform() <= p_i[1] / (mutation_rate * 100):
                ind_tmp.append(uniform(low=b_lo, high=b_up))
            else:
                ind_tmp.append(gen)
                
        return (np.array(ind_tmp), affinity(ind_tmp, prefix))
    else:
        return p_i

def select(pop, pop_clones, pop_size):
    population = pop + pop_clones
    
    population = sorted(population, key=lambda x: x[1])[:pop_size]
    
    return population

def replace(population, population_rand, population_size):
    population = population + population_rand
    population = sorted(population, key=lambda x: x[1])[:population_size]
    
    return population