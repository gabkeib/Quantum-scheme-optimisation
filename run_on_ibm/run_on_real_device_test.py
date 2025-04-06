from qiskit import QuantumCircuit, qpy
from circuits_generator import generate_circuit, get_cx_pairs, dj_function, compile_circuit, reorder_circuit, bv_function, vqc
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.least_busy(operational=True, simulator=False)

with open("bv.qpy", "rb") as qpy_file_read:
    qc = qpy.load(qpy_file_read)[0]

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)

sampler = Sampler(backend=backend)
job = sampler.run([isa_circuit])
print(f">>> Job ID: {job.job_id()}")
print(f">>> Job Status: {job.status()}")
result = job.result()
dist = result[0].data.meas.get_counts()
 
print(isa_circuit.draw())