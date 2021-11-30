import numpy as np
import random as rand
import matplotlib.pyplot as plt
import math
import time
import csv
from math import exp
from math import cos
from math import floor
from math import pi
from math import e
from math import sin
from math import sqrt


#  g(x,y) = Griewank(x,y)
def Griewank(x,y):
    f6 = ((np.power(x, 2) / 4000) + (np.power(y, 2) / 4000))  - ((np.cos(x / np.sqrt(1))) * (np.cos(y / np.sqrt(2)))) + 1
    
    return f6

def Schaffer(x, y):

    num = (np.sin(np.sqrt((x**2 + y**2)))**2) - 0.5
    den = (1 + 0.001*(x**2 + y**2))**2

    return (1 + 0.5 + num/den)

def ackley(x, y):

    return (-20 * exp(-0.2 * sqrt(0.5 * (x*x + y*y))) -
            exp(0.5 * (cos(2.0*pi*x) + cos(2*pi*y))) + e + 20)


def Rosenbrock(x, y):

    return (1 + x) ** 2 + 100 * ((x ** 2) - y) ** 2

prefix = 'ros_'

rangeMin = -20
rangeMax = 20
NP = 50
PC = 0.9
F = 0.7
D = 2
maxinteractions = 0
cont = 0
oldSumBest = 0
end = False

vectorR = np.empty((NP, D))
vectorTrial = np.empty((NP, D))
arrayResponse = []
vectorMaxBests = []
arrayBests = []
avgAcc = []
desvio_padrao = []
variancia = []

bestX = 0
bestY = 0
bestValue = 1000

responseFitnessTrial = 0
responseFitnessR = 0
count = 0
maxinteractions = 2500

# Criação da população
for i in range(NP):
    for j in range(D):
        vectorR[i][j] = rand.randint(rangeMin, rangeMax)

startTime = time.time()
# Problema de maximização
while count < maxinteractions:
    arrayResponse = []
    for i in range(NP):
        r0 = r1 = r2 = i

        while(r0 == r1 or r0 == r2 or r1 == r2 or r0 == i or r1 == i or r2 == i):
            r0 = math.floor(rand.random() * NP)
            r1 = math.floor(rand.random() * NP)
            r2 = math.floor(rand.random() * NP)

        jrand = math.floor(rand.random() * D)
        
        for j in range(D):
            if (rand.random() <= PC or j == jrand):
                vectorTrial[i][j] = vectorR[r0][j] + F * (vectorR[r1][j] - vectorR[r2][j])
            else:
                vectorTrial[i][j] = vectorR[i][j]

            vectorTrial[i][j] = np.maximum(vectorTrial[i][j], rangeMin)
            vectorTrial[i][j] = np.minimum(vectorTrial[i][j], rangeMax)

        
    # for k in range(NP):
        if prefix == 'ros_':
            responseFitnessTrial = Rosenbrock(vectorTrial[i][0], vectorTrial[i][1])
            responseFitnessR = Rosenbrock(vectorR[i][0], vectorR[i][1])
        elif prefix == 'ack_':
            responseFitnessTrial = ackley(vectorTrial[i][0], vectorTrial[i][1])
            responseFitnessR = ackley(vectorR[i][0], vectorR[i][1])

        if responseFitnessTrial < responseFitnessR:
            vectorR[i][0] = vectorTrial[i][0]
            vectorR[i][1] = vectorTrial[i][1]
            arrayResponse.append(responseFitnessTrial)
            if bestValue > responseFitnessTrial:
                bestValue = responseFitnessTrial
                bestX = vectorTrial[i][0]
                bestY = vectorTrial[i][1]
        else:
            arrayResponse.append(responseFitnessR)
            if bestValue > responseFitnessR:
                bestValue = responseFitnessTrial
                bestX = vectorR[i][0]
                bestY = vectorR[i][1]
        # responseFitnessTrial = Schaffer(vectorTrial[k][0], vectorTrial[k][1])
        # responseFitnessR = Schaffer(vectorR[k][0], vectorR[k][1])

        # if responseFitnessTrial < responseFitnessR:
        #     vectorR[k][0] = vectorTrial[k][0]
        #     vectorR[k][1] = vectorTrial[k][1]
        #     arrayResponse.append(round(responseFitnessTrial, 8))
        #     if bestValue > responseFitnessTrial:
        #         bestValue = responseFitnessTrial,
        #         bestX = vectorTrial[k][0]
        #         bestY = vectorTrial[k][1]
        # else:
        #     arrayResponse.append(round(responseFitnessR, 8))
        #     if bestValue > responseFitnessR:
        #         bestValue = responseFitnessTrial,
        #         bestX = vectorR[k][0]
        #         bestY = vectorR[k][1]

    arrayBests.append(bestValue)

    avgAcc.append(np.mean(arrayResponse))
    desvio_padrao.append(np.std(arrayResponse))
    variancia.append(np.var(arrayResponse))

    # if avgAcc[maxinteractions] - oldSumBest < 1e-6:
    #     cont += 1
        
    # if cont == 10 or maxinteractions == 10000:
    #     end = True

    # oldSumBest = avgAcc[maxinteractions]

    count += 1

endTime = time.time() 
print("Runtime: ", endTime - startTime, "s")
print("Interações: ", maxinteractions)
print("x: ", round(bestX, 8))
print("y: ", round(bestY, 8))
print("Melhor Valor: ", np.min(bestValue))
print('Média = ', np.mean(avgAcc))
plt.plot(arrayBests)
plt.plot(avgAcc)
plt.xlabel("Interações")
plt.ylabel("G(x,y)")
plt.show()


try:
    print('*********************************************')
    print('Média = ', np.mean(avgAcc))           # Média da média
    print('Menor valor = ', np.min(arrayBests))     # Menor valor do melhor indivíduo
    print('*********************************************')

    f = open(f'dados/{prefix}avgDE.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(avgAcc)
    f.close()

    f = open(f'dados/{prefix}bestDE.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(arrayBests)
    f.close()

    f = open(f'dados/{prefix}stdDE.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(desvio_padrao)
    f.close()

    f = open(f'dados/{prefix}varDE.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(variancia)
    f.close()

except:
    print("Error")