from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
import numpy as np
import random
import topologies

def generate_circuit(qubits, cnot_count, x_prob, mapping = None):
    circuit = QuantumCircuit(qubits, qubits)
    if mapping is None:
        mapping = range(qubits)
    for qubit in range(qubits):
        if random.random() < x_prob:
            circuit.x(qubit)
    for _ in range(cnot_count):
        control = random.randint(0, qubits - 1)
        target = random.randint(0, qubits - 1)
        while control == target:
            target = random.randint(0, qubits - 1)
        circuit.cx(control, target)
    circuit.measure(range(qubits), range(qubits))
    return circuit
    

# Deuthsch-Josza algorithm
def dj_function(num_qubits, mapping=None):
    if mapping is None:
        mapping = range(num_qubits + 1)

    qc = QuantumCircuit(num_qubits + 1)
    if np.random.randint(0, 2):
        qc.x(num_qubits)
    if np.random.randint(0, 2):
        return qc

    on_states = np.random.choice(
        range(2**num_qubits),
        2**num_qubits // 2,
        replace=False,
    )

    def add_cx(qc, bit_string):
        for qubit, bit in enumerate(reversed(bit_string)):
            if bit == "1":
                qc.x(qubit)
        return qc

    for state in on_states:
        qc.barrier()
        qc = add_cx(qc, f"{state:0b}")
        for qubit in range(num_qubits):
            qc.cx(qubit, num_qubits)
        qc = add_cx(qc, f"{state:0b}")

    qc.barrier()
    return qc

def get_cx_pairs(circuit):
    cx_pairs = []
    for gate in circuit:
        if gate[0].name == 'cx':
            control = gate[1][0].index
            target = gate[1][1].index
            cx_pairs.append((control, target))
    return cx_pairs


def compile_circuit(function: QuantumCircuit, mapping=None):
    n = function.num_qubits - 1
    if mapping is None:
        mapping = range(n + 1)
    qc = QuantumCircuit(n + 1, n)
    qc.x(mapping[n])
    qc.h(range(n + 1))
    qc.compose(function, inplace=True)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc

def reorder_circuit(circuit, mapping, qubits=None):
    new_circuit = QuantumCircuit(qubits if qubits is not None else circuit.num_qubits, qubits if qubits is not None else circuit.num_clbits)
    for gate in circuit:
        if gate[0].name == 'cx':
            control = gate[1][0].index
            target = gate[1][1].index
            new_circuit.cx(mapping[control], mapping[target])
        elif gate[0].name == 'measure':
            qubit = gate[1][0].index
            clbit = gate[2][0].index
            new_circuit.measure(mapping[qubit], mapping[clbit])
        else:
            new_circuit.append(gate[0], [mapping[q.index] for q in gate[1]])
    return new_circuit

def get_number_of_req_swaps(circuit, dist, mapping):
    count = 0
    for gate in circuit:
        if gate[0].name == 'cx':
            control = gate[1][0].index
            target = gate[1][1].index
            if (dist[mapping[control]][mapping[target]] != 1):
                count += (dist[mapping[control]][mapping[target]] - 1) * 2 # with return
    return count

def reorder_circuit_with_swap(circuit, mapping, dist, next, qubits=None):
    new_circuit = QuantumCircuit(qubits if qubits is not None else circuit.num_qubits, qubits if qubits is not None else circuit.num_clbits)
    required_swaps = get_number_of_req_swaps(circuit, dist, mapping)
    swaps_added = 0
    for t, gate in enumerate(circuit):
        if gate[0].name == 'cx':
            control = gate[1][0].index
            target = gate[1][1].index
            if dist[mapping[control]][mapping[target]] == 1:
                new_circuit.cx(mapping[control], mapping[target])
            else:
                get_shortest_path = topologies.construct_path(next, mapping[control], mapping[target])
                for i in range(len(get_shortest_path) - 1):
                    first_node = get_shortest_path[i]
                    second_node = get_shortest_path[i + 1]
                    if i != len(get_shortest_path) - 2:
                        new_circuit.swap(first_node, second_node)
                        first_node_index = mapping.index(first_node)
                        second_node_index = mapping.index(second_node)
                        mapping[first_node_index], mapping[second_node_index] = mapping[second_node_index], mapping[first_node_index]
                    else:
                        new_circuit.cx(first_node, second_node)
                get_shortest_path = list(reversed(get_shortest_path))
                required_swaps -= (len(get_shortest_path) - 2)
                swaps_added += (len(get_shortest_path) - 2)
                possible_required_steps = get_number_of_req_swaps(circuit[t:], dist, mapping)
                if required_swaps < possible_required_steps:
                    for i in range(1, len(get_shortest_path) - 1):
                        first_node = get_shortest_path[i]
                        second_node = get_shortest_path[i + 1]
                        new_circuit.swap(first_node, second_node)
                        first_node_index = mapping.index(first_node)
                        second_node_index = mapping.index(second_node)
                        mapping[first_node_index], mapping[second_node_index] = mapping[second_node_index], mapping[first_node_index]
                        swaps_added += 1

        elif gate[0].name == 'measure':
            qubit = gate[1][0].index
            clbit = gate[2][0].index
            new_circuit.measure(mapping[qubit], mapping[clbit])
        else:
            new_circuit.append(gate[0], [mapping[q.index] for q in gate[1]])
    return new_circuit, mapping, swaps_added

# Bernstein-Vazirani algorithm
def bv_function(num_qubits, secret_string, mapping=None):
    if mapping is None:
        mapping = range(num_qubits + 1)
    qc = QuantumCircuit(num_qubits + 1, num_qubits)
    qc.h(range(num_qubits))
    qc.x(mapping[num_qubits])
    qc.h(mapping[num_qubits])
    for qubit, bit in enumerate(reversed(secret_string)):
        if bit == "1":
            qc.cx(mapping[qubit], mapping[num_qubits])
    qc.h(range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    return qc



def grover_algorithm(num_qubits, marked_states, mapping=None):
    if mapping is None:
        mapping = range(num_qubits)
    qc = QuantumCircuit(num_qubits)
    qc.h(mapping)
    
    for state in marked_states:
        qc.x(mapping[state])
    
    oracle = QuantumCircuit(num_qubits)
    for state in marked_states:
        oracle.x(mapping[state])
    oracle.h(mapping)
    oracle.mct(mapping[:-1], mapping[-1])
    oracle.h(mapping)
    for state in marked_states:
        oracle.x(mapping[state])
    qc.append(oracle, mapping)
    
    qc.h(mapping)
    qc.x(mapping)
    qc.h(mapping[:-1])
    qc.mct(mapping[:-1], mapping[-1])
    qc.h(mapping[:-1])
    qc.x(mapping)
    qc.h(mapping)
    
    qc.measure_all()
    
    return qc



def vqc(num_qubits, nlayers=1, mapping=None):
    if mapping is None:
        mapping = range(num_qubits)
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(nlayers):
        for q in range(num_qubits):
            qc.ry(0, q)
        if num_qubits > 2:
            qc.cx(mapping[num_qubits - 1], mapping[0])
            for q in range(num_qubits-1):
                next_qubit = (q + 1) % num_qubits
                qc.cx(mapping[q], mapping[next_qubit])
        else:
            qc.cx(mapping[0], mapping[1])
    qc.measure(mapping, mapping)
    return qc


def full_vqc(num_qubits, mapping=None):
    if mapping is None:
        mapping = range(num_qubits)
    qc = QuantumCircuit(num_qubits, num_qubits)
    feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=3)
    qc.append(feature_map, range(num_qubits))
    qc.append(ansatz, range(num_qubits))
    qc.measure(mapping, mapping)
    return qc