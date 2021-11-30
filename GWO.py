import numpy as np
import matplotlib.pyplot as plt
from math import cos
from math import sqrt
from math import pi
from math import sin
from math import exp
import csv
from math import cos
from math import floor
from math import pi
from math import e
from math import sin
from math import sqrt


prefix = 'ros_'   # alterar linha 136

def sphere(x):
    return sum([item * item for item in x])

def safe_div(n,d):
    try: return n/d
    except ZeroDivisionError: return 0


class GWO:
    def __init__(self):
        self.wolf_num = 50
        self.max_iter = 2500
        self.dim = 2
        self.lb = -self.dim*np.ones((self.dim,))
        self.ub = self.dim*np.ones((self.dim,))
        self.alpha_pos = np.zeros((1,self.dim))
        self.beta_pos = np.zeros((1, self.dim))
        self.delta_pos = np.zeros((1, self.dim))
        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf
        self.convergence_curve = np.zeros((self.max_iter,))
        self.position = np.zeros((self.wolf_num,self.dim))

    def run(self):
        self.avgGWO = []
        self.desvio_padrao = []
        self.variancia = []
        count = 0
        self.init_pos()
        while count < self.max_iter:
            sumFitness = []
            for i in range(self.wolf_num):
                flag_ub = self.position[i,:] > self.ub
                flag_lb = self.position[i,:] < self.lb
                self.position[i,:] = self.position[i,:]*(~(flag_lb+flag_ub))+flag_ub*self.ub+flag_lb*self.lb
                fitness = self.func(self.position[i,:])
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.position[i,:]
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.position[i,:]
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.position[i,:]
                sumFitness.append(fitness)

            self.avgGWO.append(np.mean(sumFitness))
            self.desvio_padrao.append(np.std(sumFitness))
            self.variancia.append(np.var(sumFitness))

            a = 2 - count*(2/self.max_iter)
            for i in range(self.wolf_num):
                for j in range(self.dim):
                    alpha = self.update_pos(self.alpha_pos[j],self.position[i,j],a)
                    beta = self.update_pos(self.beta_pos[j], self.position[i, j], a)
                    delta = self.update_pos(self.delta_pos[j], self.position[i, j], a)
                    self.position[i, j] = sum(np.array([alpha, beta, delta]) * np.array([1/3,1/3,1/3]))
            count += 1
            self.convergence_curve[count-1] = self.alpha_score
        self.plot_results()
        self.save()

    def init_pos(self):
        for i in range(self.wolf_num):
            for j in range(self.dim):
                self.position[i,j] = np.random.rand()*(self.ub[j]-self.lb[j])+self.lb[j]

    @staticmethod
    def update_pos(v1,v2,a):
        A = 2*np.random.rand()*a-a
        C = 2*np.random.rand()
        temp = np.abs(C*v1-v2)
        return v1 - A*temp

    def plot_results(self):
        plt.style.use('seaborn-darkgrid')
        plt.plot(range(1,self.max_iter+1),self.convergence_curve,'g.--')
        plt.xlabel('iteration')
        plt.ylabel('fitness')
        plt.title(f'GWO fitness curve')
        plt.show()

        print(len(self.avgGWO))
        print(self.avgGWO)
        print('Média = ', np.mean(self.avgGWO))
        print('Mínimo = ', np.min(self.avgGWO))
        plt.style.use('seaborn-darkgrid')
        plt.plot(range(1,self.max_iter+1),self.avgGWO,'g.--')
        plt.xlabel('iteration')
        plt.ylabel('fitness')
        plt.title(f'GWO fitness curve')
        plt.show()

    def save(self):
        try:
            print('*********************************************')
            print('Média = ', np.mean(self.avgGWO))  # Média da média
            print('Menor valor = ', np.min(self.convergence_curve))  # Menor valor do melhor indivíduo
            print('*********************************************')

            f = open(f'dados/{prefix}bestGWO.csv', 'a', newline='', encoding='utf-8')
            w = csv.writer(f, delimiter=";")
            w.writerow(self.convergence_curve)
            f.close()

            f = open(f'dados/{prefix}avgGWO.csv', 'a', newline='', encoding='utf-8')
            w = csv.writer(f, delimiter=";")
            w.writerow(self.avgGWO)
            f.close()

            f = open(f'dados/{prefix}stdGWO.csv', 'a', newline='', encoding='utf-8')
            w = csv.writer(f, delimiter=";")
            w.writerow(self.desvio_padrao)
            f.close()

            f = open(f'dados/{prefix}varGWO.csv', 'a', newline='', encoding='utf-8')
            w = csv.writer(f, delimiter=";")
            w.writerow(self.variancia)
            f.close()

        except:
            print("Error")
    '''
    @staticmethod
    def func(input): # Schaffer    
        [x, y] = input
        num = (np.sin(np.sqrt((x**2 + y**2)))**2) - 0.5
        den = (1 + 0.001*(x**2 + y**2))**2 

        return (1 + 0.5 + num/den)
        
    @staticmethod
    def func(input):  # rastrigin
    x = input
    for item in x:
        assert x<=5.12 and x>=-5.12, 'input exceeds bounds of [-5.12, 5.12]'
    return len(x)*10.0 +  sum([item*item - 10.0*cos(2.0*pi*item) for item in x])
    
    @staticmethod
    def func(input): # Ackley
    [x, y] = input
    return (-20 * exp(-0.2 * sqrt(0.5 * (x*x + y*y))) -
            exp(0.5 * (cos(2.0*pi*x) + cos(2*pi*y))) + e + 20)

    @staticmethod
    def func(x):  # Rosenbrock
        total = 0
        for i in range(len(x) - 1):
            total += 100 * (x[i + 1] - x[i] * x[i]) ** 2 + (x[i] - 1) ** 2
        return total  
    '''
    @staticmethod
    def func(x):  # Rosenbrock
        total = 0
        for i in range(len(x) - 1):
            total += 100 * (x[i + 1] - x[i] * x[i]) ** 2 + (x[i] - 1) ** 2
        return total

if __name__ == "__main__":
    gwo = GWO()
    gwo.run()