from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService
from convert_result import convert_result
service = QiskitRuntimeService(channel="ibm_quantum")

from qiskit.visualization import plot_histogram

job = service.job('aaaaaaaaaaaaaaaa')
job_result = job.result()

best_permutation = []

def parity(x):
    return "{:b}".format(x).count("1") % 2

def parity2(x):
    return x % 2


for idx, pub_result in enumerate(job_result):
    answer_length = 9

    parity_res = {
        '0': 0,
        '1': 0
    }

    res = {
        
    }

    # for vqc
    # res = {
    #     '0000': 0,
    #     '0001': 0,
    #     '0010': 0,
    #     '0011': 0,
    #     '0100': 0,
    #     '0101': 0,
    #     '0110': 0,
    #     '0111': 0,
    #     '1000': 0,
    #     '1001': 0,  
    #     '1010': 0,
    #     '1011': 0,
    #     '1100': 0,
    #     '1101': 0,
    #     '1110': 0,
    #     '1111': 0,
    # }

    for key, value in pub_result.data.c.get_counts().items():
        get_reduced = convert_result(string=key, mapping=best_permutation, qubits_topology=127, answer_length=answer_length, crop_answer=True)
        if get_reduced not in res:
            res[get_reduced] = value
        else:
            res[get_reduced] += value
        
        # for vqc
        # parity_res[str(parity(int(get_reduced, 2)))] += value
        # print(int(get_reduced, 2), parity2(int(get_reduced, 2)))
        # parity_res[str(parity2(int(get_reduced, 2)))] += value

    # for vqc
    # parity_res["0"] = parity_res["0"] / 4096
    # parity_res["1"] = parity_res["1"] / 4096
    # fig = plot_histogram([parity_res])
    # ax = fig.axes[0]
    # ax.set_xlabel("Class")
    # ax.set_ylabel("Probability")
    # fig.savefig("./results/vqc.png", bbox_inches="tight")

    fig = plot_histogram([res], figsize=(20, 6))
    
    ax = fig.axes[0]
    ax.set_xlabel("Vector")
    ax.set_ylabel("Count")
    fig.savefig("./results/bv.png", bbox_inches="tight")