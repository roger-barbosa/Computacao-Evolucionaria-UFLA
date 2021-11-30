import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

numberRound = 6

prefix = 'ros_'   # prefixo: ros_ = Rosenbrock - sch_ = Schaffer


# PSO
bestPSO = []
with open(f"dados/{prefix}bestPSO.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            bestPSO.append(round(float(j),numberRound))
file.close()

avgPSO = []
with open(f"dados/{prefix}avgPSO.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            avgPSO.append(round(float(j),numberRound))

file.close()

varPSO = []
with open(f"dados/{prefix}varPSO.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            varPSO.append(round(float(j),numberRound))

file.close()

stdPSO = []
with open(f"dados/{prefix}stdPSO.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            stdPSO.append(round(float(j),numberRound))

file.close()

# DE
avgDE = []
with open(f"dados/{prefix}avgDE.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            avgDE.append(round(float(j),numberRound))
file.close()

bestDE = []
with open(f"dados/{prefix}bestDE.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            bestDE.append(round(float(j),numberRound))

file.close()

stdDE = []
with open(f"dados/{prefix}stdDE.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            stdDE.append(round(float(j),numberRound))

file.close()

varDE = []
with open(f"dados/{prefix}varDE.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            varDE.append(round(float(j),numberRound))

file.close()


# AG
avgAG = []
with open(f"dados/{prefix}avgAG.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            avgAG.append(round(float(j),numberRound))
file.close()

bestsAG = []
with open(f"dados/{prefix}bestAG.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            bestsAG.append(round(float(j),numberRound))

file.close()

varAG = []
with open(f"dados/{prefix}varAG.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            x = round(float(j),numberRound)
            #print(f"dados/{prefix}varAG.csv", x)
            varAG.append(x)

file.close()

stdAG = []
with open(f"dados/{prefix}stdAG.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            x = round(float(j),numberRound)
            #print(f"dados/{prefix}stdAG.csv", x)
            stdAG.append(x)

file.close()


# ALO
bestsALO = []
with open(f"dados/{prefix}bestALO.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            bestsALO.append(round(float(j),numberRound))

file.close()

avgALO = []
with open(f"dados/{prefix}avgALO.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            avgALO.append(round(float(j),numberRound))

file.close()

varALO = []
with open(f"dados/{prefix}varALO.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            varALO.append(round(float(j),numberRound))

file.close()

stdALO = []
with open(f"dados/{prefix}stdALO.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            stdALO.append(round(float(j),numberRound))

file.close()

# CLONALG
avgCLONALG = []
with open(f"dados/{prefix}avgCLONALG.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            avgCLONALG.append(round(float(j),numberRound))

file.close()

bestsCLONALG = []
with open(f"dados/{prefix}bestCLONALG.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            bestsCLONALG.append(round(float(j),numberRound))

file.close()

varCLONALG = []
with open(f"dados/{prefix}varCLONALG.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            varCLONALG.append(round(float(j),numberRound))

file.close()

stdCLONALG = []
with open(f"dados/{prefix}stdCLONALG.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            stdCLONALG.append(round(float(j),numberRound))

file.close()


# GWO
bestsGWO = []
with open(f"dados/{prefix}bestGWO.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            bestsGWO.append(round(float(j),numberRound))

file.close()

avgGWO = []
with open(f"dados/{prefix}avgGWO.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            avgGWO.append(round(float(j),numberRound))

file.close()

stdGWO = []
with open(f"dados/{prefix}stdGWO.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            stdGWO.append(round(float(j),numberRound))

file.close()

varGWO = []
with open(f"dados/{prefix}varGWO.csv", encoding='utf-8') as file:
    table = csv.reader(file, delimiter=";")
    for l in table:
        for j in l:
            varGWO.append(round(float(j),numberRound))

file.close()

# fig, (ax1, ax2) = plt.subplots(2, 1)
# fig.subplots_adjust(hspace=0.5)
# ax1.plot(bestPSO)
# ax1.plot(bestDE)
# ax1.plot(bestsGWO)
# ax1.set_xlabel('Interações')
# ax1.set_ylabel('Best Value')
# ax1.legend(['PSO', 'DE', 'GWO'])
# ax1.grid(True)

# ax2.plot(bestsAG)
# ax2.plot(avgCLONALG)
# ax2.plot(bestsALO)
# ax2.set_xlabel('Interações')
# ax2.set_ylabel('Best Value')
# ax2.legend(['AG', 'ALO', 'CLONALG'])
# ax2.grid(True)
# plt.show()

plt.plot(avgPSO)
plt.plot(avgDE)
plt.plot(avgAG)
plt.plot(avgALO)
plt.plot(avgGWO)
plt.plot(avgCLONALG)
if prefix == 'ack_':
    plt.title('Função Ackley')
else:
    plt.title('Função Rosenbrock')
plt.xlabel('Interações')
plt.ylabel('Média')
#plt.legend(['AG'])
plt.legend(['PSO', 'DE', 'AG', 'ALO', 'GWO', 'CLONALG'])
plt.grid(True)
plt.show()

if prefix == 'ack_':
    plt.title('Função Ackley')
else:
    plt.title('Função Rosenbrock')
plt.plot(bestPSO)
plt.plot(bestDE)
plt.plot(bestsAG)
plt.plot(bestsALO)
plt.plot(bestsGWO)
plt.plot(bestsCLONALG)
plt.xlabel('Interações')
plt.ylabel('Melhor Valor')
#plt.legend(['AG'])
plt.legend(['PSO', 'DE', 'AG', 'ALO', 'GWO', 'CLONALG'])
plt.grid(True)
plt.show()

'''
if prefix == 'ack_':
    plt.title('Função Ackley')
else:
    plt.title('Função Rosenbrock')
plt.plot(stdPSO)
plt.plot(stdDE)
plt.plot(stdAG)
plt.plot(stdALO)
plt.plot(stdGWO)
plt.plot(stdCLONALG)
plt.xlabel('Interações')
plt.ylabel('Desvio Padrão')
plt.legend(['PSO', 'DE', 'AG', 'ALO', 'GWO', 'CLONALG'])
#plt.grid(True)
plt.show()

if prefix == 'ack_':
    plt.title('Função Ackley')
else:
    plt.title('Função Rosenbrock')
plt.plot(varPSO)
plt.plot(varDE)
plt.plot(varAG)
plt.plot(varALO)
plt.plot(varGWO)
plt.plot(varCLONALG)
plt.xlabel('Interações')
plt.ylabel('Variância')
plt.legend(['PSO', 'DE', 'AG', 'ALO', 'GWO', 'CLONALG'])
#plt.grid(True)
plt.show()

'''

