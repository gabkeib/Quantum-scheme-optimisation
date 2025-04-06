def convert_result(string, mapping, qubits_topology, answer_length, crop_answer=False):
    updated_string = string[::-1]
    new_key = [0 for _ in range(qubits_topology)]
    for i in range(qubits_topology):
        new_key[mapping[i]] = updated_string[i]
    new_key = ''.join(new_key)
    new_key = new_key[:answer_length] if crop_answer else new_key
    return new_key
