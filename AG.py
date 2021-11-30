from math import exp
from math import cos
from math import pi
from math import e
from math import sqrt

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time
import csv
#%matplotlib inline


def criação_intervalo(input_variaveis):  # Array com intervalos inferior, superior e número de bits

    input_variaveis = np.asmatrix(input_variaveis)
    intervalos = []
    intervalos_invertidos = []

    for variavel in input_variaveis:
        max_intervalo = float(np.asarray(variavel[:, :2]).max())
        min_intervalo = float(np.asarray(variavel[:, :2]).min())
        n_bits = int(variavel[:, 2:])

        # Criação de bins

        tamanho_intervalo = max_intervalo - min_intervalo + 1
        qtd_numeros_representaveis = 2 ** n_bits
        tamanho_bin = (tamanho_intervalo) / qtd_numeros_representaveis
        lista_bins = np.linspace(min_intervalo, max_intervalo, qtd_numeros_representaveis).tolist()

        # Criação de lista de conversor decimal-binario:

        lista_binarios = []
        formato_binario = (str('#0' + str(n_bits + 2) + 'b'))
        i = 0
        while i < qtd_numeros_representaveis:
            lista_binarios.append(format(i, formato_binario))
            i = i + 1

        intervalo = dict(zip(lista_bins, lista_binarios))
        intervalo_invertido = {v: k for k, v in intervalo.items()}

        intervalos.append(intervalo)
        intervalos_invertidos.append(intervalo_invertido)

    return intervalos, intervalos_invertidos


def conversor_dec_bin(input_valores, input_dicionarios):
    valores_convertidos = []

    for dicionario, linha in enumerate(input_valores):
        array = [*input_dicionarios[dicionario]]
        max_intervalo = max(array)
        min_intervalo = min(array)

        array = np.asarray(array)

        valores_convertidos_por_variavel = []

        for valor in linha:

            if (valor > max_intervalo) or (valor < min_intervalo):
                raise ValueError('Erro. Valor fora do intervalo')

            idx = (np.abs(array - valor)).argmin()

            valores_convertidos_por_variavel.append(input_dicionarios[dicionario].get(array[idx]))

        valores_convertidos.append(valores_convertidos_por_variavel)

    return valores_convertidos


def conversor_bin_dec(input_binarios, input_dicionarios):
    input_dicionarios = intervalos_invertidos

    binarios_convertidos = []

    for idx, linha in enumerate(input_binarios):

        binarios_convertidos_por_variavel = []

        for binario in linha:
            formato_binario = str('#0' + str(len([*input_dicionarios[idx]][1])) + 'b')
            binario = int(str(binario), 2)
            binario = format(binario, formato_binario)

            if (2 ** (len(binario) - 2)) > (len(input_dicionarios[idx])):
                raise ValueError('Erro. Valor fora do intervalo')
            else:
                binarios_convertidos_por_variavel.append(input_dicionarios[idx].get(binario))

        binarios_convertidos.append(binarios_convertidos_por_variavel)

    return binarios_convertidos


def conversor_bin_vetor(input_strings_binarias):
    vetores_binarios = []

    for linha in input_strings_binarias:
        vetores_binarios_por_variavel = []

        for string_binaria in linha:
            vetores_binarios_por_variavel.append([int(caractere) for caractere in str(string_binaria)[2:]])

        vetores_binarios.append(vetores_binarios_por_variavel)

    return vetores_binarios


def conversor_vetor_bin(input_vetores_binario):
    valores_binarios = []

    for linha in input_vetores_binario:
        valores_binarios_por_linha = []
        for valor in linha:
            valores_binarios_por_linha.append(('0b' + ''.join(str(bit) for bit in valor)))

        valores_binarios.append(valores_binarios_por_linha)

    return valores_binarios

def decimal2binario(input_vetores_decimais, input_intervalos):
    strings_binarias = conversor_dec_bin(input_vetores_decimais, input_intervalos)
    vetores_binarios = conversor_bin_vetor(strings_binarias)
    return vetores_binarios

def binario2decimal(vetores_binarios, input_intervalos_invertidos):
    strings_binarias = conversor_vetor_bin(vetores_binarios)
    valores_decimais = conversor_bin_dec(strings_binarias, input_intervalos_invertidos)
    return valores_decimais

def transposta(input_matrix):
    matrix_array = np.asarray(input_matrix)
    return matrix_array.T


# Função geral: gera vetor transposto da população e aplica função selecionada

def funções(input_função, input_pop_decimal):
    pop_decimal_transposta = transposta(input_pop_decimal)

    return input_função(pop_decimal_transposta)


def flat_pop(input_pop):
    n_individuo = 0
    pop_flat = []

    while n_individuo < len(input_pop[0]):
        individuo = []
        n_variavel = 0
        while n_variavel < len(input_pop):
            for bit in (input_pop[n_variavel][n_individuo]):
                individuo.append(bit)
            n_variavel = n_variavel + 1

        n_individuo = n_individuo + 1
        pop_flat.append(individuo)

    return pop_flat

def unflat_pop(input_pop_aux):

    unflat_pop = []
    unflat_variavel = []

    for idx_individuo, individuo in enumerate(input_pop_aux):
        count_acum = 0
        for idx, variavel in enumerate(vetor_n_bits):
            unflat_variavel = []
            count_aux = 0
            while count_aux < variavel:
                unflat_variavel.append(input_pop_aux[idx_individuo][count_acum])
                count_aux = count_aux + 1
                count_acum = count_acum + 1

            unflat_pop.append(unflat_variavel)

    output_pop = []

    for idx, variavel in enumerate(vetor_n_bits):
        armazenamento_por_variavel = []
        count = idx
        while count < len(unflat_pop):
            armazenamento_por_variavel.append(unflat_pop[count])
            count = count + len(vetor_n_bits)

        output_pop.append(armazenamento_por_variavel)

    return output_pop


def plot_resultados(resultados, media_resultados):
    # Resultado Geral
    plt.plot(resultados, 'blue')
    plt.style.use('seaborn-darkgrid')
    plt.title('Resultado Geral')
    plt.xlabel('Geração')
    plt.ylabel('f(x)')
    plt.legend(['Melhor Performance da Geração'])
    plt.rcParams['figure.figsize'] = (20, 10)

    # plt.figure()
    plt.show()

    # Resultado Médio
    plt.plot(media_resultados, 'black')
    plt.style.use('seaborn-darkgrid')
    plt.title('Resultado Médio')
    plt.xlabel('Geração')
    plt.ylabel('f(x)')
    plt.legend(['Performance Média da Geração'])
    plt.rcParams['figure.figsize'] = (20, 10)

    # plt.figure()
    plt.show()


def iniciar_população(input_n_individuos, input_vetor_n_bits, input_seed):
    random.seed(input_seed)
    pop = []
    n_var = len(input_vetor_n_bits)

    for linha in range(0, input_n_individuos):
        individuo = []
        for coluna in range(0, sum(input_vetor_n_bits)):
            aleatorio = random.uniform(0, 1)
            individuo.append(round(aleatorio))
        pop.append(individuo)

    # Divide a população da seguinte forma:
    # pop_por_variavel = [[indiviuos_variavel_1], [indiviuos_variavel_2], [indiviuos_variavel_3], [indiviuos_variavel_n]]

    bit_inicio = 0
    bit_termino = 0
    pop_por_variavel = []

    for idx, n_bits in enumerate(input_vetor_n_bits):
        # Separação da população por variáveis

        bit_termino = bit_inicio + n_bits
        pop_por_variavel.append((np.asarray(pop)[:, bit_inicio:bit_termino]).tolist())
        bit_inicio = bit_inicio + n_bits

    return pop_por_variavel

def seleção(input_método_seleção, input_fx, input_tipo_otimização):
    return input_método_seleção(input_fx, input_tipo_otimização)


def método_seleção_roleta(input_fx, input_tipo_otimização):
    roleta_soma = sum(input_fx)
    roleta_soma = np.array(roleta_soma)
    roleta_média = np.mean(input_fx)
    roleta_máximo = np.max(input_fx)

    if input_tipo_otimização == 'max':  # Se max
        fi_sobre_roleta_soma = input_fx / roleta_soma
        prob_se_max = fi_sobre_roleta_soma
        roleta_probabilidade_acum = np.cumsum(prob_se_max)
    else:  # Se min
        prob_se_min = ((roleta_soma / input_fx) / (sum(roleta_soma / input_fx)))
        roleta_probabilidade_acum = np.cumsum(prob_se_min)

    índices_pais = []

    # Performa os trials:
    n_giro_roleta = 0
    while n_giro_roleta < n_individuos:
        idx = (np.abs(roleta_probabilidade_acum - random.uniform(0, 1))).argmin()
        índices_pais.append(idx)
        n_giro_roleta = n_giro_roleta + 1

    return índices_pais


def método_seleção_roleta_log(input_fx, input_tipo_otimização):
    roleta_soma = sum(np.log(input_fx))
    roleta_soma = np.array(roleta_soma)
    roleta_média = np.mean(np.log(input_fx))
    roleta_máximo = np.max(np.log(input_fx))

    if input_tipo_otimização == 'max':  # Se max
        fi_sobre_roleta_soma = np.log(input_fx) / roleta_soma
        prob_se_max = fi_sobre_roleta_soma
        roleta_probabilidade_acum = np.cumsum(prob_se_max)
    else:  # Se min
        prob_se_min = ((roleta_soma / np.log(input_fx)) / (sum(roleta_soma / np.log(input_fx))))
        roleta_probabilidade_acum = np.cumsum(prob_se_min)

    índices_pais = []

    # Performa os trials:
    n_giro_roleta = 0
    while n_giro_roleta < n_individuos:
        idx = (np.abs(roleta_probabilidade_acum - random.uniform(0, 1))).argmin()
        índices_pais.append(idx)
        n_giro_roleta = n_giro_roleta + 1

    return índices_pais


def método_seleção_torneio(input_fx, input_tipo_otimização):
    # Gera dois arrays com números aleatórios
    # Compara qual indivíduo tem a melhor performance conforme tipo de otimização

    pais = np.random.randint(len(input_fx), size=(2, len(input_fx)))
    idx_a = pais[0, :]
    idx_b = pais[1, :]
    fx_a = np.array(input_fx)[idx_a.astype(int)]
    fx_b = np.array(input_fx)[idx_b.astype(int)]

    # se max:
    if input_tipo_otimização == 'max':
        selecionados = np.array(fx_a) > np.array(fx_b)
    # se min:
    else:
        selecionados = np.array(fx_a) < np.array(fx_b)

    índices_pais = []
    for idx, item in enumerate(selecionados):
        if item == True:
            índices_pais.append(idx_a[idx])
        else:
            índices_pais.append(idx_b[idx])

    return índices_pais

def cruzamento(input_método_cruzamento, input_índices_pop_aux, input_pop, input_probabilidade_cruzamento):
    return input_método_cruzamento(input_índices_pop_aux, input_pop, input_probabilidade_cruzamento)


def cruzamento_um_ponto(input_índices_pop_aux, input_pop, input_probabilidade_cruzamento):
    pais = np.random.randint(len(input_índices_pop_aux), size=(2, int(len(input_índices_pop_aux) / 2)))

    idx_a = pais[0, :]
    idx_b = pais[1, :]

    pop_matriz = flat_pop(input_pop)

    len_individuo = len(pop_matriz[0])

    pop_aux = []
    cruzamento = 0

    while cruzamento < (len(idx_a)):

        # Definições do cruzamento
        ponto_cruzamento = int(random.uniform(1,
                                              len_individuo))  # Sorteio de ponto de corte. Na nossa implementação, cada cruzamento tem um ponto de corte diferente
        probabilidade_cruzamento = random.uniform(0,
                                                  1)  # Se maior ou igual que a input_probabilidade_cruzamento, define se haverá ou não cruzamento

        # Define pais
        pai_a = pop_matriz[idx_a[cruzamento]]
        pai_b = pop_matriz[idx_b[cruzamento]]

        novo_individuo = []

        # POSSIBILIDADE A: Se houver cruzamento
        if input_probabilidade_cruzamento >= probabilidade_cruzamento:

            # Executa o cruzamento
            filho_a = pai_a[:ponto_cruzamento] + pai_b[ponto_cruzamento:]
            filho_b = pai_b[:ponto_cruzamento] + pai_a[ponto_cruzamento:]

            pop_aux.append(filho_a)
            pop_aux.append(filho_b)

        # POSSIBILIDADE B: Se não houver cruzamento passa os pais adiante
        else:

            pop_aux.append(pai_a)
            pop_aux.append(pai_b)

        cruzamento = cruzamento + 1
    return pop_aux

def mutação(input_método_mutação, input_pop_aux, input_probabilidade_mutação):
    return input_método_mutação(input_pop_aux, input_probabilidade_mutação)


def mutação_um_ponto(input_pop_aux, input_probabilidade_mutação):
    # Seleciona quantidade de indivíduos igual à probabilidade de mutação (1)
    # Seleciona bit aleatório de variável aleatória (2)
    # Inverte bit (3)
    # Insere indivíduo na população (4)

    # (1)
    n_mutações = 0
    len_indivíduo = len(pop_aux[0])

    while n_mutações < round(input_probabilidade_mutação * len(pop_aux)):

        # (2)
        índice_indivíduo_aleatório = int(random.uniform(0, len(pop_aux)))

        # Seleção do bit aleatório
        posição_bit_aleatório = int(random.uniform(0, len_indivíduo))

        valor_bit_aleatório = pop_aux[índice_indivíduo_aleatório][posição_bit_aleatório]

        # (3)
        if valor_bit_aleatório == 0:
            novo_bit = 1
        else:
            novo_bit = 0

        # (4)
        pop_aux[índice_indivíduo_aleatório][posição_bit_aleatório] = novo_bit

        n_mutações = n_mutações + 1

    return pop_aux


def elitismo(input_fx_elite, input_pop_matriz_elite, input_fx_aux, input_pop_aux, input_tipo_otimização,
             input_qtd_individuos_elitismo):
    # f(x) elite
    concat_fx = input_fx_elite + input_fx_aux
    índices_elite_aux = np.argsort(concat_fx)

    concat_pop = input_pop_matriz_elite + input_pop_aux

    if input_tipo_otimização == 'max':
        índices_elite_final = índices_elite_aux[-int(len(concat_pop) - input_qtd_individuos_elitismo):]

    else:
        índices_elite_final = índices_elite_aux[:int(len(concat_pop) - input_qtd_individuos_elitismo)]

    pop_matriz_final = []

    for índice_elite_final in índices_elite_final:
        pop_matriz_final.append(concat_pop[índice_elite_final])

    return pop_matriz_final

def função_schaffer(input_pop_decimal_transposta):
    y = []

    for individuo in input_pop_decimal_transposta:
        parentesis = 0
        for xi in individuo:
            parentesis = parentesis + xi ** 2

        num = (np.sin(np.sqrt((parentesis)))**2) - 0.5
        dem = (1 + 0.001*(parentesis))**2
        schaffer = 1 + 0.5 + num / dem
        
        y.append(schaffer)
    return y

# Função Rosenbrock
'''
def funcao_rosenbrock(input_pop_decimal_transposta):
    y = []
    total = 0
    for i in range(len(input_pop_decimal_transposta) - 1):
        a = input_pop_decimal_transposta[i + 1]
        b = input_pop_decimal_transposta[i]

        total += 100 * (input_pop_decimal_transposta[i + 1] - input_pop_decimal_transposta[i] *
                        input_pop_decimal_transposta[i]) ** 2 + (input_pop_decimal_transposta[i] - 1) ** 2
        y.append(total)
    return total
'''
def funcao_rosenbrock(input_pop_decimal_transposta):
    y = []
    for individuo in input_pop_decimal_transposta:
        rosenbrock = 0
        for idx, xi in enumerate(individuo):
            if idx == (len(individuo) - 1):
                break
            rosenbrock = rosenbrock + (100 * (((xi ** 2) - individuo[idx + 1]) ** 2) + ((xi - 1) ** 2))
        y.append(rosenbrock)

    return y


def funcao_rastrigin(input_pop_decimal_transposta, safe_mode=False):
    y = []
    x = input_pop_decimal_transposta
    if safe_mode:
        for item in x:
            assert x<=5.12 and x>=-5.12, 'input exceeds bounds of [-5.12, 5.12]'
    for individuo in input_pop_decimal_transposta:
        rosenbrock = 0
        for idx, xi in enumerate(individuo):
            if idx == (len(individuo) - 1):
                break
            y.append(len(x)*10.0 + sum([xi*xi - 10.0*cos(2.0*pi*xi) for xi in x]))

    return y

def funcao_ackley(input_pop_decimal_transposta):
    y = []

    for individuo in input_pop_decimal_transposta:

        ackley = 0
        soma_1 = 0
        soma_2 = 0

        for xi in individuo:
            soma_1 = soma_1 + xi ** 2
            soma_2 = soma_2 + math.cos(2 * math.pi * xi)

        termo_1 = -20 * math.exp(-0.2 * (math.sqrt(1 / len(individuo) * soma_1)))
        termo_2 = - math.exp((1 / len(individuo)) * soma_2)

        ackley = termo_1 + termo_2 + 20 + math.exp(1)
        y.append(ackley)

    return y

# Intervalo e número de bits por variável:

variaveis = [[-20, 20, 10],[-20, 20, 10]]
intervalos, intervalos_invertidos = criação_intervalo(variaveis)

# Parâmetros

vetor_n_bits = np.asarray(variaveis)[:, 2].astype(int)
n_individuos = 50
seed = 30
input_método_seleção = método_seleção_torneio
input_tipo_otimização = 'min'
probabilidade_cruzamento = .8
input_método_cruzamento = cruzamento_um_ponto
input_método_mutação = mutação_um_ponto
input_probabilidade_mutação = 0.05
input_função = funcao_rosenbrock  #  função_schaffer  #  funcao_rosenbrock

input_qtd_individuos_elitismo = int(.05*n_individuos)

n_gen = 0
n_max_gen = 2500

#% % time

media_resultados = []
resultados = []
desvio_padrao = []
variancia = []
# Evolução

pop = iniciar_população(n_individuos, vetor_n_bits, seed)
pop_decimal = binario2decimal(pop, intervalos_invertidos)
fx_aux = funções(input_função, pop_decimal)

avgAg = []
startTime = time.time()
while n_gen < n_max_gen:

    # (1) Backup do resultado anterior (melhores indivíduos) para elitismo

    # Captura dos índices e resultados com melhor desempenho
    if input_tipo_otimização == 'max':
        fx_elite = sorted(fx_aux)[-int(input_qtd_individuos_elitismo):]
        índices_elite = np.argsort(fx_aux)[-int(input_qtd_individuos_elitismo):]

    else:
        fx_elite = sorted(fx_aux)[:int(input_qtd_individuos_elitismo)]
        índices_elite = np.argsort(fx_aux)[:int(input_qtd_individuos_elitismo)]

    # Captura dos elementos da população com melhor desempenho
    backup_pop_matriz = flat_pop(pop)
    backup_pop_matriz_elite = []

    for índice_elite in índices_elite:
        backup_pop_matriz_elite.append(backup_pop_matriz[índice_elite])

        # (2) Seleção
    índices_pop_aux = seleção(input_método_seleção, fx_aux, input_tipo_otimização)

    # (3) Cruzamento
    pop_aux = cruzamento(input_método_cruzamento, índices_pop_aux, pop, probabilidade_cruzamento)

    # (4) Mutação
    pop_aux = mutação(input_método_mutação, pop_aux, input_probabilidade_mutação)

    # (5) Avaliação Parcial, pré-elitismo
    pop = unflat_pop(pop_aux)
    pop_decimal = binario2decimal(pop, intervalos_invertidos)
    fx_aux = funções(input_função, pop_decimal)

    # (6) Elitismo
    pop_matriz_final = elitismo(fx_elite, backup_pop_matriz_elite, fx_aux, pop_aux, input_tipo_otimização,
                                input_qtd_individuos_elitismo)

    # (5) Avaliação Final, pós-elitismo
    pop = unflat_pop(pop_matriz_final)
    pop_decimal = binario2decimal(pop, intervalos_invertidos)
    fx_aux = funções(input_função, pop_decimal)

    # Resultados
    media_resultados.append(np.mean(fx_aux))
    desvio_padrao.append(np.std(fx_aux))  # Desvio padão
    variancia.append(np.var(fx_aux))      #   Variancia

    if input_tipo_otimização == 'max':
        resultados.append(max(fx_aux))
        if n_gen == (n_max_gen - 1):
            print('O valor máximo obtido foi', max(fx_aux))
            print('Os valores numéricos de entrada que trouxeram esse resultado foram',
                  transposta(pop_decimal)[np.argmax(fx_aux)])
    else:
        resultados.append(min(fx_aux))
        if n_gen == (n_max_gen - 1):
            print('O valor mínimo obtido foi', min(fx_aux))
            print('Os valores numéricos de entrada que trouxeram esse resultado foram',
                  transposta(pop_decimal)[np.argmin(fx_aux)])

    n_gen = n_gen + 1

endTime = time.time()  

print("Runtime: ", endTime - startTime, "s")
print()
plot_resultados(resultados, media_resultados)

#plt.plot(media_resultados)   # Calcular
#plt.xlabel('Interações')
#plt.ylabel('Best Value')
#plt.show()



try:
    if input_função == funcao_rosenbrock:
        prefix = 'ros_'
    elif input_função == funcao_ackley:
        prefix = 'ack_'

    print('*********************************************')
    print('Média = ', np.mean(media_resultados))  # média
    print('Menor valor = ', np.min(resultados))   # Menor valor
    print('*********************************************')


    f = open(f'dados/{prefix}avgAG.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(media_resultados)
    f.close()

    f = open(f'dados/{prefix}bestAG.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(resultados)
    f.close()


    f = open(f'dados/{prefix}stdAG.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(desvio_padrao)
    f.close()

    f = open(f'dados/{prefix}varAG.csv', 'a', newline='', encoding='utf-8')
    w = csv.writer(f, delimiter=";")
    w.writerow(variancia)
    f.close()


except:
    print("Error")