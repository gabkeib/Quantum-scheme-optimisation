from qiskit import QuantumCircuit, execute, Aer, qpy
from collections import deque
from optimisation import simulated_annealing
from circuits_generator import generate_circuit, get_cx_pairs, dj_function, compile_circuit, reorder_circuit, bv_function, vqc
from optim_utils import generate_random_binary_string
import topologies
import numpy as np
import argparse
from distutils.util import strtobool
import datetime

def add_two_qubit_cnot_gate(circuit, qubit1, qubit2):
    circuit.cx(qubit1, qubit2)
    graph.append([qubit1, qubit2])

def add_pauli_x_gates(circuit, qubit_list):
    for qubit in qubit_list:
        circuit.x(qubit)

def generate_adjacency_matrix(graph):
    graph_matrix = [[[0 for _ in range(qubits)] for _ in range(qubits)]]
    for edge in graph:
            graph_matrix[len(graph_matrix) - 1][edge[0]][edge[1]] = 1
            graph_matrix[len(graph_matrix) - 1][edge[1]][edge[0]] = 1
    return np.array(graph_matrix)

def prepare_topology(topology, n):
    return np.array([topology for _ in range(n)])

def get_mapping(permutation):
    mapping = [0 for _ in range(len(permutation))]
    for i, x in enumerate(permutation):
        mapping[x] = i
    return mapping

def prepare_circ(circuit, permutation):
    circ = np.zeros((len(circuit), len(permutation), len(permutation)))
    mapping = get_mapping(permutation)
    for i in range(len(circuit)):
        lst = []
        for j, perm in enumerate(circuit[i]):
            for t in range(len(perm)):
                if circuit[i][j][t] == 1:
                    lst.append((mapping[j], mapping[t]))
        for (j, t) in lst:
            circ[i][j][t] = 1
    return circ 

def objective_function(permutation):
    circ = prepare_circ(circuit_matrix, permutation)
    return -np.sum(np.multiply(circ, topology))

def read_arguments():
    parser = argparse.ArgumentParser(description='Optimise a quantum circuit')
    parser.add_argument('--qubits', type=int, help='Number of qubits in the circuit', default=3, nargs='?')
    parser.add_argument('--test', type=strtobool, help='Whether to test the result', default=True, nargs='?')
    parser.add_argument('--crop_answer', type=bool, help='Whether to crop the answer', default=True, nargs='?')
    parser.add_argument('--only_ones', type=bool, help='Whether to generate only 1s', default=False, nargs='?')
    parser.add_argument('--generate_string', type=bool, help='Whether to generate a string', default=True, nargs='?')
    parser.add_argument('--topology', type=str, help='Computer topology to use', default='ionq_harmony_q9', nargs='?')
    parser.add_argument('--debug', type=strtobool, help='Debug mode', default=False, nargs='?')
    return parser.parse_args()


args = read_arguments()
qubits = args.qubits
test = args.test
crop_answer = args.crop_answer
only_ones = args.only_ones
generate_string = args.generate_string
topology_name = args.topology
debug = args.debug

graph = []    
pauli_x_gates = [2]

# Prepare quantum circuit
if only_ones:
    bv_string = [1 for _ in range(qubits)]
else:
    bv_string = generate_random_binary_string(qubits) if generate_string else '101'

if debug:
    print('BV string:', bv_string)

circuit = vqc(qubits)
qubits += 1
graph = get_cx_pairs(circuit)

if debug:
    print(graph)
    for gate in circuit.data:
        print('\ngate name:', gate[0].name)
        print('qubit(s) acted on:', gate[1])
        print('other paramters (such as angles):', gate[0].params)

    print(get_cx_pairs(circuit))
    print(circuit)

if test:

    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1000)

    result = job.result()
    counts = result.get_counts(circuit)
    if debug:
        print(counts)

# Prepare optimization
circuit_matrix = generate_adjacency_matrix(graph)
req_edges = np.sum(circuit_matrix)
topology = topologies.get_topology_by_string(topology_name)
topology = prepare_topology(topology, len(circuit_matrix))

qubits_topology = len(topology[0])

if qubits_topology < qubits:
    raise ValueError("Topology matrix is too small")

circuit_matrix_2 = np.array([[
    [0 for _ in range(qubits_topology)] for _ in range(qubits_topology)
] for _ in range(len(circuit_matrix))])

for i in range(len(circuit_matrix)):
    for j in range(qubits):
        for t in range(qubits):
            circuit_matrix_2[i][j][t] = circuit_matrix[i][j][t]

circuit_matrix = circuit_matrix_2

# Simulated annealing
num_elements = qubits_topology
num_iterations = 1000
initial_temperature = 100
cooling_rate = 0.95 

start_time = datetime.datetime.now()
found_mapping = False

max_iterations = 10

while not found_mapping and max_iterations > 0:
    best_permutation, best_score = simulated_annealing(objective_function, num_elements, num_iterations, initial_temperature, cooling_rate)
    found_mapping = True if best_score * (-1) == len(graph) * 2 else False
    max_iterations -= 1

end_time = datetime.datetime.now()
duration = end_time - start_time
duration_in_s = duration.total_seconds()

if debug:
    print("Best permutation:", best_permutation)
    print("Best score:", best_score)
    print('Required:', req_edges)

# Test the result
if test:
    mapping = get_mapping(best_permutation)
    circ2 = reorder_circuit(circuit, mapping=mapping, qubits=qubits_topology)
    
    if debug:
        print(circ2)
        print(best_permutation)

    job2 = execute(circ2, simulator, shots=1000)

    result2 = job2.result()
    counts2 = result2.get_counts(circ2)
    
    mapping = best_permutation

    if debug:
        print(counts2)
        print(mapping)

    key = list(counts2.keys())[0][::-1]
    new_key = [0 for _ in range(qubits_topology)]
    for i in range(qubits_topology):
        new_key[mapping[i]] = key[i]
    new_key = ''.join(new_key)
    new_key = new_key[:qubits - 1] if crop_answer else new_key


    old_key = list(counts.keys())[0][::-1]

    if debug:
        print(''.join(new_key))
        print(list(counts.keys())[0][::-1])
        print('Same: ', ''.join(new_key) == list(counts.keys())[0][::-1])


if debug:
    print('Found mapping:', found_mapping)

with open(f'./Results/vc/{topology_name}.csv', 'a') as f:
    f.write(f'{bv_string},{found_mapping},{duration_in_s}\n')








