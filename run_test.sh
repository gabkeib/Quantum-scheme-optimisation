#!/bin/bash

declare -a topologies=("ibm_reuschlikon_q16" "ibm_tokyo_q20" "ibm_paughkeepsie_q20" "ibm_cambridge_q28" "ibm_montreal_q27" "ibm_almaden_q20" "ibm_rochester_q53" "ibm_falcon_q27" "rigetti_novera_q9" "ionq_harmony_q9" "ibm_manhattan_q65" "ibm_eagle_q127" "ibm_heron_q133")

declare -a qubitscount=(16 20 20 28 27 20 53 27 9 9 65 127 133)

N=25
pos=0
start=2

for i in "${topologies[@]}"
do
    echo "Running on ${i}"
    for qb in $(eval echo {$start..${qubitscount[$pos]}})
    do
        for x in $(eval echo {1..$N})
        do
            # python3 qiskit_to_with_swap.py --topology $i --test=False --qubits=$qb --expand_circuit=True
            python3 qiskit_to.py --topology $i --test=False --qubits=$qb
        done
    done
    pos=$((pos+1))
done
