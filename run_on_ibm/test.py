from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate
from qiskit.visualization import plot_distribution
from circuits_generator import generate_circuit, get_cx_pairs, dj_function, compile_circuit, bv_function, vqc, reorder_circuit_with_swap
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
import topologies
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, qpy
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import RealAmplitudes
import datetime
from qiskit_machine_learning.algorithms.classifiers import VQC
# from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA, ADAM
import json
import numpy as np
# from qiskit import IBMQ

def get_mapping(permutation):
    mapping = [0 for _ in range(len(permutation))]
    for i, x in enumerate(permutation):
        mapping[x] = i
    return mapping

def reorder_circuit(circuit, mapping, qubits=None):
    new_circuit = QuantumCircuit(qubits if qubits is not None else circuit.num_qubits, qubits if qubits is not None else circuit.num_clbits)
    for gate in circuit:
        print(gate)
        if gate[0].name == 'cx':
            control = gate[1][0].index
            target = gate[1][1].index
            new_circuit.cx(mapping[control], mapping[target])
        elif gate[0].name == 'measure':
            qubit = gate[1][0].index
            clbit = gate[2][0].index
            new_circuit.measure(mapping[qubit], mapping[clbit])
        else:
            new_circuit.append(gate.operation, [mapping[q._index] for q in gate.qubits])
    return new_circuit


def cost_func_vqe(params, ansatz, hamiltonian, estimator):
    cost = estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    return cost

def preprocess_forward(
        input_data: np.ndarray | None,
        weights: np.ndarray | None,
    ) -> tuple[np.ndarray | None, int | None]:
        if input_data is not None:
            num_samples = input_data.shape[0]
            if weights is not None:
                weights = np.broadcast_to(weights, (num_samples, len(weights)))
                parameters = np.concatenate((input_data, weights), axis=1)
            else:
                parameters = input_data
        else:
            if weights is not None:
                num_samples = 1
                parameters = np.broadcast_to(weights, (num_samples, len(weights)))
            else:
                num_samples = 1
                parameters = np.asarray([])
        return parameters, num_samples

with open('data.json') as f:
    d = json.load(f)

features = np.array(d['features'])
labels = np.array(d['labels'])

# Example permutation
best_permutation = [45, 116, 103, 99, 17, 41, 19, 37, 57, 120, 6, 8, 78, 81, 22, 86, 16, 50, 15, 84, 91, 70, 69, 60, 3, 13, 113, 65, 39, 9, 29, 77, 88, 96, 2, 75, 42, 80, 89, 35, 98, 112, 109, 0, 1, 44, 83, 71, 68, 33, 27, 124, 118, 56, 104, 12, 20, 38, 92, 125, 36, 43, 48, 5, 31, 94, 24, 18, 123, 21, 30, 58, 63, 25, 73, 46, 82, 54, 40, 122, 101, 102, 66, 10, 32, 28, 114, 76, 23, 108, 64, 93, 126, 74, 95, 105, 34, 87, 97, 67, 52, 72, 121, 4, 61, 111, 110, 59, 26, 53, 85, 11, 55, 51, 62, 115, 100, 119, 47, 79, 7, 90, 14, 49, 106, 107, 117]

features = MinMaxScaler().fit_transform(features)


algorithm_globals.random_seed = 123
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=algorithm_globals.random_seed
)

initial_weights = [0.3336368,   1.13438416, -0.10508152,  1.67057472, -0.07756552,  2.93083464,
  0.47836086,  0.87711967,  0.22056006, -0.31703886,  1.88508948,  1.73756833,
  1.14302921,  0.33876296,  0.74146705,  1.44162241]

parameters = list(test_features[0]) + list(initial_weights)
print(parameters)

params, num_samples = preprocess_forward(input_data=np.array([features[0]]), weights=initial_weights)
print(params, num_samples)

num_features = features.shape[1]
mapping = get_mapping(best_permutation)


feature_map = ZFeatureMap(feature_dimension=num_features, reps=1)
feature_map = reorder_circuit(feature_map, mapping=mapping, qubits=127)

service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.least_busy(operational=True, simulator=False)
print(backend.name)
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

sampler = Sampler(backend=backend, options={'max_execution_time': 15})
sampler.options.max_execution_time = 15

topology = topologies.get_topology_by_string('ibm_eagle_q127')
dist, next = topologies.get_all_shortest_paths(topology)

ansatz = RealAmplitudes(num_qubits=2, entanglement='circular', reps=3)
ansatz, mapping, swaps = reorder_circuit_with_swap(ansatz, mapping=mapping, dist=dist, next=next, qubits=127)

# print(ansatz.draw())
# ansatz = pm.run(ansatz)

circuit = QuantumCircuit(127)
circuit.compose(feature_map, inplace=True)
circuit.compose(ansatz, inplace=True)
circuit.measure_all()

circuit = pm.run(circuit)
