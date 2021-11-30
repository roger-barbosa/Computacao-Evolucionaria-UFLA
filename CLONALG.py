import matplotlib.pyplot as plt
import seaborn as sns

from clonalg_code import clonalg
from pprint import pprint
import numpy as np
import csv

# Inputs parameters
prefix = 'ros_'

b_lo, b_up = (-20, 20)

population_size = 50
selection_size = 10
problem_size = 2
random_cells_num = 20
clone_rate = 0.8
mutation_rate = 0.2
stop_codition = 2500

stop = 0

# Population <- CreateRandomCells(Population_size, Problem_size)
population = clonalg.create_random_cells(population_size, problem_size, b_lo, b_up)
# print(population)
# a = b
best_affinity_it = []
arrayBest = []
desvio_padrao = []
variancia = []

while stop != stop_codition:
    # Affinity(p_i)
    population_affinity = [(p_i, clonalg.affinity(p_i, prefix)) for p_i in population]
    populatin_affinity = sorted(population_affinity, key=lambda x: x[1])
    
    best_affinity_it.append(populatin_affinity[:50])
    
    # Populatin_select <- Select(Population, Selection_size)
    population_select = populatin_affinity[:selection_size]
    
    # Population_clones <- clone(p_i, Clone_rate)
    population_clones = []
    for p_i in population_select:
        p_i_clones = clonalg.clone(p_i, clone_rate)
        population_clones += p_i_clones
        
    # Hypermutate and affinity
    pop_clones_tmp = []
    for p_i in population_clones:
        ind_tmp = clonalg.hypermutate(p_i, mutation_rate, b_lo, b_up, prefix)
        pop_clones_tmp.append(ind_tmp)
    population_clones = pop_clones_tmp
    del pop_clones_tmp
    
    # Population <- Select(Population, Population_clones, Population_size)
    population = clonalg.select(populatin_affinity, population_clones, population_size)
    # Population_rand <- CreateRandomCells(RandomCells_num)
    population_rand = clonalg.create_random_cells(random_cells_num, problem_size, b_lo, b_up)
    population_rand_affinity = [(p_i, clonalg.affinity(p_i, prefix)) for p_i in population_rand]
    # print(population_rand_affinity)
    # arrayBest.append(np.min(np.array(population_rand_affinity)[:,1]))
    population_rand_affinity = sorted(population_rand_affinity, key=lambda x: x[1])
    # Replace(Population, Population_rand)
    population = clonalg.replace(population_affinity, population_rand_affinity, population_size)
    population = [p_i[0] for p_i in population]
    
    stop += 1

# We get the mean of the best 5 individuals returned by iteration of the above loop
bests_mean = []
bests_min = []
iterations = [i for i in range(2500)]

for pop_it in best_affinity_it:
    bests_mean.append(np.mean([p_i[1] for p_i in pop_it]))
    bests_min.append(np.min([p_i[1] for p_i in pop_it]))
    desvio_padrao.append(np.std([p_i[1] for p_i in pop_it]))
    variancia.append(np.var([p_i[1] for p_i in pop_it]))

print('bests_mean (média) ', np.mean(bests_mean), bests_mean)
print('bests_min (minimo) ', np.min(bests_min), bests_min)


fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi=150)

sns.set_style("darkgrid")
sns.pointplot(x=iterations, y=bests_mean)
sns.pointplot(x=iterations, y=bests_min)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title("Mean of 5 Best Individuals by Iteration", fontsize=12)
plt.ylabel("Affinity Mean", fontsize=10)
plt.rc('ytick',labelsize=2)
plt.xlabel("# Iteration", fontsize=10)
plt.show()

try:
    print('*********************************************')
    print('Média = ', np.mean(bests_mean))           # Média da média
    print('Menor valor = ', np.min(bests_min))     # Menor valor do melhor indivíduo
    print('*********************************************')

    f = open(f'dados/{prefix}bestCLONALG.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(bests_mean)
    f.close()

    f = open(f'dados/{prefix}avgCLONALG.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(bests_min)
    f.close()

    f = open(f'dados/{prefix}stdCLONALG.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(desvio_padrao)
    f.close()

    f = open(f'dados/{prefix}varCLONALG.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(variancia)
    f.close()


except:
    print("Error")