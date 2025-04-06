import numpy as np
import random
import argparse

def generate_random_coupled_matrix(qubits, p = 0.5, zeros_negative = False):
    adj_matrix = np.full((qubits, qubits), -1) if zeros_negative else np.zeros((qubits, qubits))
    for i in range(qubits):
        for j in range(qubits):
            if j > i:
                if random.random() < p:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
    return adj_matrix

def convert_to_binary(matrix, p = 0.5):
    n = len(matrix)
    m = len(matrix[0])
    for i in range(n):
        for j in range(m):
            if matrix[i][j] >= p:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
    return matrix

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-n', default=5)
    args = parser.parse_args()
    return int(args.n)  

def generate_random_binary_string(n):
    b_string = [random.choice([0, 1]) for _ in range(n)]
    while sum(b_string) == 0:
        b_string = [random.choice([0, 1]) for _ in range(n)]
    return ''.join([str(x) for x in b_string])