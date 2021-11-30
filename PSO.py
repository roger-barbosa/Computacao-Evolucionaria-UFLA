# Versao de James D. McCaffrey
# https://jamesmccaffrey.wordpress.com/2015/06/09/particle-swarm-optimization-using-python/
# particleswarm.py
# python 3.4.3
# demo of particle swarm optimization (PSO)
# solves Rastrigin's function

import random
import math    # cos() for Rastrigin
import copy    # array-copying convenience
import sys     # max float
import numpy as np
import matplotlib.pyplot as plt
import csv
import time as time
from math import exp
from math import cos
from math import floor
from math import pi
from math import e
from math import sin
from math import sqrt


# ------------------------------------

prefix = 'ros_'   # 'ros_'  ou 'sch_'
tempoI = time.time()
def rastrigin(x, safe_mode=False):
    '''Rastrigin Function

    Parameters
    ----------
        x : list
        safe_mode : bool (optional, default = False)

    Returns
    -------
        float

    Notes
    -----
    Bounds: -5.12 <= x_i <= 5.12 for all i=1,...,d
    Global minimum: f(x)=0 at x=(0,...,0)

    References
    ----------
    wikipedia: https://en.wikipedia.org/wiki/Rastrigin_function
    '''

    if safe_mode:
        for item in x:
            assert x<=5.12 and x>=-5.12, 'input exceeds bounds of [-5.12, 5.12]'
    return len(x)*10.0 +  sum([item*item - 10.0*cos(2.0*pi*item) for item in x])

def ackley(xy):
    '''
    Ackley Function

    wikipedia: https://en.wikipedia.org/wiki/Ackley_function

    global minium at f(x=0, y=0) = 0
    bounds: -5<=x,y<=5
    '''
    x, y = xy[0], xy[1]
    return (-20 * exp(-0.2 * sqrt(0.5 * (x*x + y*y))) -
            exp(0.5 * (cos(2.0*pi*x) + cos(2*pi*y))) + e + 20)


def show_vector(vector):
  for i in range(len(vector)):
    if i % 8 == 0: # 8 columns
      print("\n", end="")
    if vector[i] >= 0.0:
      print(' ', end="")
    print("%.4f" % vector[i], end="") # 4 decimals
    print(" ", end="")
  print("\n")

def error(position):
    if prefix == 'sch_':
        [x,y] = position
        num = (np.sin(np.sqrt((x**2 + y**2)))**2) - 0.5
        den = (1 + 0.001*(x**2 + y**2))**2
        ret = (1 + 0.5 + num/den)
    elif prefix == 'ros_':
        x = position
        ret = 0
        for i in range(len(x) - 1):
            ret += 100 * (x[i + 1] - x[i] * x[i]) ** 2 + (x[i] - 1) ** 2
    elif prefix == 'ras_':
        ret = rastrigin(position)
    elif prefix == 'ack_':
        ret = ackley(position)

    return ret

# ------------------------------------

class Particle:
  def __init__(self, dim, minx, maxx, seed):
    self.rnd = random.Random(seed)
    self.position = [0.0 for i in range(dim)]
    self.velocity = [0.0 for i in range(dim)]
    self.best_part_pos = [0.0 for i in range(dim)]

    for i in range(dim):
      self.position[i] = ((maxx - minx) *
        self.rnd.random() + minx)
      self.velocity[i] = ((maxx - minx) *
        self.rnd.random() + minx)

    self.error = error(self.position) # curr error
    self.best_part_pos = copy.copy(self.position) 
    self.best_part_err = self.error # best error

bestPSO = []
avgPSO = []
desvio_padrao = []
variancia = []

swarErrors = []
def Solve(max_epochs, n, dim, minx, maxx):
  rnd = random.Random(0)

  # create n random particles
  swarm = [Particle(dim, minx, maxx, i) for i in range(n)] 

  best_swarm_pos = [0.0 for i in range(dim)] # not necess.
  best_swarm_err = sys.float_info.max # swarm best
  for i in range(n): # check each particle
    if swarm[i].error < best_swarm_err:
      best_swarm_err = swarm[i].error
      best_swarm_pos = copy.copy(swarm[i].position) 

  epoch = 0
  w = 0.729    # inertia
  c1 = 2.05  # cognitive (particle)   fator cognitivo
  c2 = 2.05  # social (swarm)   fator social
  fc = 0.73
  #tempoI = time.time()
  while epoch < max_epochs:

    for i in range(n): # process each particle
        # compute new velocity of curr particle
        for k in range(dim): 
            r1 = rnd.random()    # randomizations
            r2 = rnd.random()
        
            swarm[i].velocity[k] = fc * ( ( w * swarm[i].velocity[k]) +
            (c1 * r1 * (swarm[i].best_part_pos[k] -
            swarm[i].position[k])) +  
            (c2 * r2 * (best_swarm_pos[k] -
            swarm[i].position[k])) )  

            if swarm[i].velocity[k] < minx:
                swarm[i].velocity[k] = minx
            elif swarm[i].velocity[k] > maxx:
                swarm[i].velocity[k] = maxx

        # compute new position using new velocity
        for k in range(dim): 
            swarm[i].position[k] += swarm[i].velocity[k]
    
        # compute error of new position
        swarm[i].error = error(swarm[i].position)

        # is new position a new best for the particle?
        if swarm[i].error < swarm[i].best_part_err:
            swarm[i].best_part_err = swarm[i].error
            swarm[i].best_part_pos = copy.copy(swarm[i].position)

        # is new position a new best overall?
        if swarm[i].error < best_swarm_err:
            best_swarm_err = swarm[i].error
            best_swarm_pos = copy.copy(swarm[i].position)
        swarErrors.append(swarm[i].error)
    
    avgPSO.append(np.mean(swarErrors))
    bestPSO.append(best_swarm_err)
    desvio_padrao.append(np.std(swarErrors))
    variancia.append(np.var(swarErrors))

    w = 0.9 - epoch * ((0.9 - 0.4) / max_epochs)
    # for-each particle
    epoch += 1
  # while
  return best_swarm_pos
# end Solve

print('Tempo de execução = ', time.time()-tempoI)

print("\nBegin particle swarm optimization using Python demo\n")
dim = 2
print("Goal is to solve Rastrigin's function in " +
 str(dim) + " variables")
print("Function has known min = 1.0 at (", end="")
for i in range(dim-1):
  print("0, ", end="")
print("0)")

num_particles = 50
max_epochs = 2500

print("Setting num_particles = " + str(num_particles))
print("Setting max_epochs    = " + str(max_epochs))
print("\nStarting PSO algorithm\n")

best_position = Solve(max_epochs, num_particles, dim, -20.0, 20.0)

print("\nPSO completed\n")
print("\nBest solution found:")
show_vector(best_position)
err = error(best_position)
print("Error of best solution = %.6f" % err)

print("\nEnd particle swarm demo\n")

print('MÉDIA = ', avgPSO)


plt.plot(avgPSO)
plt.plot(bestPSO)
plt.show()


try:
    print('*********************************************')
    print('Média = ', np.mean(avgPSO))           # Média da média
    print('Menor valor = ', np.min(bestPSO))     # Menor valor do melhor indivíduo
    print('*********************************************')

    f = open(f'dados/{prefix}avgPSO.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(avgPSO)
    f.close()

    f = open(f'dados/{prefix}bestPSO.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(bestPSO)
    f.close()

    f = open(f'dados/{prefix}stdPSO.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(desvio_padrao)
    f.close()

    f = open(f'dados/{prefix}varPSO.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(variancia)
    f.close()

except:
    print("Error")