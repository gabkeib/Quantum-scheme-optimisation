from qiskit import QuantumCircuit, execute, Aer, qpy
from collections import deque
from optimisation import simulated_annealing
from circuits_generator import generate_circuit, get_cx_pairs, dj_function, compile_circuit, reorder_circuit, bv_function, vqc, reorder_circuit_with_swap
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
        if graph_matrix[len(graph_matrix) - 1][edge[0]][edge[1]] == 0 and graph_matrix[len(graph_matrix) - 1][edge[1]][edge[0]] == 0:
            graph_matrix[len(graph_matrix) - 1][edge[0]][edge[1]] = 1
            graph_matrix[len(graph_matrix) - 1][edge[1]][edge[0]] = 1
        else:
            graph_matrix.append([[0 for _ in range(qubits)] for _ in range(qubits)])
            graph_matrix[len(graph_matrix) - 1][edge[0]][edge[1]] = 1
            graph_matrix[len(graph_matrix) - 1][edge[1]][edge[0]] = 1
    return np.array(graph_matrix)


def prepare_topology(topology, n):
    return np.array([topology for _ in range(n)])

def all_good(circuit, dist):
    for gate in circuit:
        if gate[0].name == 'cx':
            control = gate[1][0].index
            target = gate[1][1].index
            if dist[control][target] != 1:
                return False
    return True

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
    mapping = get_mapping(permutation)
    sum = 0
    for gate in circuit:
        if gate[0].name == 'cx':
            control = gate[1][0].index
            target = gate[1][1].index
            sum += dist[mapping[control]][mapping[target]]
    return sum


def read_arguments():
    parser = argparse.ArgumentParser(description='Optimise a quantum circuit')
    parser.add_argument('--qubits', type=int, help='Number of qubits in the circuit', default=3, nargs='?')
    parser.add_argument('--test', type=strtobool, help='Whether to test the result', default=True, nargs='?')
    parser.add_argument('--crop_answer', type=bool, help='Whether to crop the answer', default=True, nargs='?')
    parser.add_argument('--only_ones', type=bool, help='Whether to generate only 1s', default=False, nargs='?')
    parser.add_argument('--generate_string', type=bool, help='Whether to generate a string', default=True, nargs='?')
    parser.add_argument('--topology', type=str, help='Computer topology to use', default='ionq_harmony_q9', nargs='?')
    parser.add_argument('--debug', type=strtobool, help='Debug mode', default=False, nargs='?')
    parser.add_argument('--expand_circuit', type=strtobool, help='Whether to expand circuit using swaps', default=False, nargs='?')
    return parser.parse_args()


args = read_arguments()
qubits = args.qubits
test = args.test
crop_answer = args.crop_answer
only_ones = args.only_ones
generate_string = args.generate_string
topology_name = args.topology
debug = args.debug
expand_circuit = args.expand_circuit


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


topology = topologies.get_topology_by_string(topology_name)
dist, next = topologies.get_all_shortest_paths(topology)
topology = prepare_topology(topology, len(circuit_matrix))

# Resize the circuit matrix to match the number of qubits
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
best_permutation = None
best_score = 1e6

while not found_mapping and max_iterations > 0:
    best_permutation_curr, best_score_curr = simulated_annealing(objective_function, num_elements, num_iterations, initial_temperature, cooling_rate)
    if best_score > best_score_curr:
        best_permutation = best_permutation_curr
        best_score = best_score_curr 
    found_mapping = True if best_score == len(graph) else False
    max_iterations -= 1



if expand_circuit:
    mapping = get_mapping(best_permutation)
    circ2, mapping, swaps_added = reorder_circuit_with_swap(circuit, mapping=mapping, dist=dist, next=next, qubits=qubits_topology)



end_time = datetime.datetime.now()
duration = end_time - start_time
duration_in_s = duration.total_seconds()

if debug:
    print("Best permutation:", best_permutation)
    print("Best score:", best_score)

# Test the result
if test:
    if debug:
        print(circ2)
        print(best_permutation)
    
    job2 = execute(circ2, simulator, shots=1000)

    if swaps_added > 0:
        with open("vc_2.qpy", "wb") as qpy_file_write:
            qpy.dump(circ2, qpy_file_write)


    result2 = job2.result()
    counts2 = result2.get_counts(circ2)
    mapping = get_mapping(mapping)

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
    print('All good?', all_good(circ2, dist))
    print('Swaps added', swaps_added)

with open(f'./Results/bv/{topology_name}.csv', 'a') as f:
    if expand_circuit:
        f.write(f'{bv_string},{found_mapping},{duration_in_s},{swaps_added},{all_good(circ2, dist)}\n')
    else:
        f.write(f'{bv_string},{found_mapping},{duration_in_s}\n')








