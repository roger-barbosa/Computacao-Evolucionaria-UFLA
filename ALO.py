############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Ant Lion Optimizer

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Ant_Lion_Optimizer, File: Python-MH-Ant_Lion_Optimizer.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Ant_Lion_Optimizer>

############################################################################

# Required Libraries
import numpy  as np
import math
import random
import os
import time
import csv
import matplotlib.pyplot as plt
from math import exp
from math import cos
from math import floor
from math import pi
from math import e
from math import sin
from math import sqrt

# Function
def target_function():
    return

# Function: Initialize Variables
def initial_population(colony_size = 50, min_values = [-20,-20], max_values = [20,20], target_function = target_function):
    population = np.zeros((colony_size, len(min_values) + 1))
    for i in range(0, colony_size):
        for j in range(0, len(min_values)):
             population[i,j] = random.uniform(min_values[j], max_values[j]) 
        population[i,-1] = target_function(population[i,0:population.shape[1]-1])
    
    return population

# Function: Fitness
def fitness_function(population): 
    fitness = np.zeros((population.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1+ population[i,-1] + abs(population[:,-1].min()))
    fit_sum = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix = 0
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix

# Function: Random Walk
def random_walk(iterations):
    x_random_walk = [0]*(iterations + 1)
    x_random_walk[0] = 0
    for k in range(1, len( x_random_walk)):
        rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        if rand > 0.5:
            rand = 1
        else:
            rand = 0
        x_random_walk[k] = x_random_walk[k-1] + (2*rand - 1)       
    return x_random_walk

# Function: Combine Ants
def combine(population, antlions):
    combination = np.vstack([population, antlions])
    combination = combination[combination[:,-1].argsort()]
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[1]):
            antlions[i,j]   = combination[i,j]
            population[i,j] = combination[i + population.shape[0],j]
    return population, antlions

# Function: Update Antlion
def update_ants(population, antlions, count, iterations, min_values = [-20,-20], max_values = [20,20], target_function = target_function):
    i_ratio       = 1
    minimum_c_i   = np.zeros((1, population.shape[1]))
    maximum_d_i   = np.zeros((1, population.shape[1]))
    minimum_c_e   = np.zeros((1, population.shape[1]))
    maximum_d_e   = np.zeros((1, population.shape[1]))
    elite_antlion = np.zeros((1, population.shape[1]))
    if  (count > 0.10*iterations):
        w_exploration = 2
        i_ratio = (10**w_exploration)*(count/iterations)  
    elif(count > 0.50*iterations):
        w_exploration = 3
        i_ratio = (10**w_exploration)*(count/iterations)   
    elif(count > 0.75*iterations):
        w_exploration = 4
        i_ratio = (10**w_exploration)*(count/iterations)    
    elif(count > 0.90*iterations):
        w_exploration = 5
        i_ratio = (10**w_exploration)*(count/iterations)   
    elif(count > 0.95*iterations):
        w_exploration = 6
        i_ratio = (10**w_exploration)*(count/iterations)
    for i in range (0, population.shape[0]):
        fitness = fitness_function(antlions)
        ant_lion = roulette_wheel(fitness)
        for j in range (0, population.shape[1] - 1):   
            minimum_c_i[0,j]   = antlions[antlions[:,-1].argsort()][0,j]/i_ratio
            maximum_d_i[0,j]   = antlions[antlions[:,-1].argsort()][-1,j]/i_ratio
            elite_antlion[0,j] = antlions[antlions[:,-1].argsort()][0,j]
            minimum_c_e[0,j]   = antlions[antlions[:,-1].argsort()][0,j]/i_ratio
            maximum_d_e[0,j]   = antlions[antlions[:,-1].argsort()][-1,j]/i_ratio  
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (rand < 0.5):
                minimum_c_i[0,j] =   minimum_c_i[0,j] + antlions[ant_lion,j]
                minimum_c_e[0,j] =   minimum_c_e[0,j] + elite_antlion[0,j]
            else:
                minimum_c_i[0,j] = - minimum_c_i[0,j] + antlions[ant_lion,j]
                minimum_c_e[0,j] = - minimum_c_e[0,j] + elite_antlion[0,j]
                
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (rand >= 0.5):
                maximum_d_i[0,j] =   maximum_d_i[0,j] + antlions[ant_lion,j]
                maximum_d_e[0,j] =   maximum_d_e[0,j] + elite_antlion[0,j]
            else:
                maximum_d_i[0,j] = - maximum_d_i[0,j] + antlions[ant_lion,j]
                maximum_d_e[0,j] = - maximum_d_e[0,j] + elite_antlion[0,j]   
            x_random_walk = random_walk(iterations)
            e_random_walk = random_walk(iterations)    
            min_x, max_x = min(x_random_walk), max(x_random_walk)
            x_random_walk[count] = (((x_random_walk[count] - min_x)*(maximum_d_i[0,j] - minimum_c_i[0,j]))/(max_x - min_x)) + minimum_c_i[0,j]   
            min_e, max_e = min(e_random_walk), max(e_random_walk)
            e_random_walk[count] = (((e_random_walk[count] - min_e)*(maximum_d_e[0,j] - minimum_c_e[0,j]))/(max_e - min_e)) + minimum_c_e[0,j]    
            population[i,j] = np.clip((x_random_walk[count] + e_random_walk[count])/2, min_values[j], max_values[j])
        population[i,-1] = target_function(population[i,0:population.shape[1]-1])
        return population, antlions

arrayBest = []
avgALO = []
desvio_padrao = []
variancia = []

# ALO Function
def ant_lion_optimizer(colony_size = 50, min_values = [-20,-20], max_values = [20,20], iterations = 5000, target_function = target_function):    
    count = 0  
    population = initial_population(colony_size = colony_size, min_values = min_values, max_values = max_values, target_function = target_function)
    antlions   = initial_population(colony_size = colony_size, min_values = min_values, max_values = max_values, target_function = target_function) 
    elite = np.copy(antlions[antlions[:,-1].argsort()][0,:]) 
    startTime = time.time()

    while (count <= iterations):
        # print("Iteration = ", count, " f(x) = ", elite[-1])   
        population, antlions = update_ants(population, antlions, count = count, iterations = iterations, min_values = min_values, max_values = max_values, target_function = target_function)
        population, antlions = combine(population, antlions)    
        value = np.copy(antlions[antlions[:,-1].argsort()][0,:])
        avgALO.append(np.mean(antlions[:,-1]))
        arrayBest.append(value[-1])
        desvio_padrao.append(np.std(antlions[:,-1]))
        variancia.append(np.var(antlions[:,-1]))
        if(elite[-1] > value[-1]):
            elite = np.copy(value)
        else:
            antlions[antlions[:,-1].argsort()][0,:] = np.copy(elite)   
        count = count + 1 

    endTime = time.time() 

    print("Runtime: ", endTime - startTime, "s")
    print("Iteration = ", count, " f(x) = ", elite[-1])
    print(elite, np.mean(elite))
    return elite

######################## Part 1 - Usage ####################################

# Function to be Minimized (Easom). Solution ->  f(x) = -1; xi = 3.14
def easom(variables_values = [0, 0]):
    return -math.cos(variables_values[0])*math.cos(variables_values[1])*math.exp(-(variables_values[0] - math.pi)**2 - (variables_values[1] - math.pi)**2)

# alo = ant_lion_optimizer(colony_size = 80, min_values = [-1,-1], max_values = [7,7], iterations = 100, target_function = easom)

# Function to be Minimized (Six Hump Camel Back). Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def six_hump_camel_back(variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

# alo = ant_lion_optimizer(colony_size = 80, min_values = [-5,-5], max_values = [5,5], iterations = 500, target_function = six_hump_camel_back)

# Function to be Minimized (Rosenbrocks Valley). Solution ->  f(x) = 0; xi = 1
def rosenbrocks_valley(variables_values = [0,0]):
    func_value = 0
    last_x = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + (100 * math.pow((variables_values[i] - math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)
    return func_value

def Rosenbrock(x):
    total = 0
    for i in range(len(x)-1):
        total += 100*(x[i+1] - x[i]*x[i])**2 + (x[i]-1)**2
    return total


#  g(x,y) = 1 + Schaffer(x,y)
def Schaffer(variables_values = [0,0]):
    num = (np.sin(np.sqrt((variables_values[0]**2 + variables_values[1]**2)))**2) - 0.5
    den = (1 + 0.001*(variables_values[0]**2 + variables_values[1]**2))**2 
    func_value = 1 + 0.5 + num/den
    return  func_value

def rastrigin(x, safe_mode=False):

    if safe_mode:
        for item in x:
            assert x<=5.12 and x>=-5.12, 'input exceeds bounds of [-5.12, 5.12]'
    return len(x)*10.0 +  sum([item*item - 10.0*cos(2.0*pi*item) for item in x])

def ackley(xy):
    x, y = xy[0], xy[1]
    ret = (-20 * exp(-0.2 * sqrt(0.5 * (x*x + y*y))) -
            exp(0.5 * (cos(2.0*pi*x) + cos(2*pi*y))) + e + 20)
    return ret


alo = ant_lion_optimizer(colony_size = 50, min_values = [-20,-20], max_values = [20,20], iterations = 2500,
                         target_function = Rosenbrock)

try:
    funcao = 'ROS'
    if funcao == 'ROS':
        prefix = 'ros_'
    elif funcao == 'SCH':
        prefix = 'sch_'
    elif funcao == 'RAS':
        prefix = 'ras_'
    elif funcao == 'ACK':
        prefix = 'ack_'

    print('*********************************************')
    print('Média = ', np.mean(avgALO))           # Média da média
    print('Menor valor = ', np.min(arrayBest))   # Menor valor do melhor indivíduo
    print('*********************************************')


    f = open(f'dados/{prefix}bestALO.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(arrayBest)
    f.close()

    f = open(f'dados/{prefix}avgALO.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(avgALO)
    f.close()

    f = open(f'dados/{prefix}stdALO.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(desvio_padrao)
    f.close()

    f = open(f'dados/{prefix}varALO.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(variancia)
    f.close()

except:
    print("Error")