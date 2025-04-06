import random
import numpy as np 

def prepare_circ(circuit, permutation):
    circ = np.zeros((len(circuit), len(permutation), len(permutation)))
    mapping = [0 for _ in range(len(permutation))]
    for i, x in enumerate(permutation):
        mapping[x] = i
    for i in range(len(circuit)):
        lst = []
        for j, perm in enumerate(circuit[i]):
            for t in range(len(perm)):
                if circuit[i][j][t] == 1:
                    lst.append((mapping[j], mapping[t]))
        for (j, t) in lst:
            circ[i][j][t] = 1
    return circ 


num_elements = 6
num_iterations = 50
initial_temperature = 10
cooling_rate = 0.95


def simulated_annealing(obj_func, num_elements, num_iterations, initial_temperature, cooling_rate):
    current_permutation = list(range(num_elements))
    random.shuffle(current_permutation)
    
    best_permutation = current_permutation
    best_score = obj_func(current_permutation)
    
    temperature = initial_temperature
    for _ in range(num_iterations):
        # Generate neighbor permutation
        neighbor_permutation = current_permutation[:]
        # Swap two random elements
        idx1, idx2 = random.sample(range(num_elements), 2)
        neighbor_permutation[idx1], neighbor_permutation[idx2] = neighbor_permutation[idx2], neighbor_permutation[idx1]
        
        current_score = obj_func(current_permutation)
        neighbor_score = obj_func(neighbor_permutation)
        
        # Decide whether to move to neighbor
        if neighbor_score < current_score or random.random() < acceptance_probability(current_score, neighbor_score, temperature):
            current_permutation = neighbor_permutation[:]
            current_score = neighbor_score
            
            if current_score < best_score:
                best_permutation = current_permutation[:]
                best_score = current_score
        
        # Cool down temperature
        temperature *= cooling_rate
    
    return best_permutation, best_score

def acceptance_probability(current_score, neighbor_score, temperature):
    if neighbor_score < current_score:
        return 1.0
    else:
        return pow(2.71828, -(neighbor_score - current_score) / temperature)
