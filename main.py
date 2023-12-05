""""
https://qiskit.org/ecosystem/aer/tutorials/2_device_noise_simulation.html

possible objetive function: f(E, S, A) = k_1*E + k_2*S - k_3*A


"""

from qclib.gates.mcu import MCU
from utilities import Utilities as Ut
from qiskit import (QuantumCircuit,
                    QuantumRegister,
                    ClassicalRegister,
                    transpile)
from qiskit.visualization import (plot_histogram,
                                  circuit_drawer)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeVigo
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import time


def built_circuit(pauli_string='x', error=0.1):
    """
    Builds a quantum circuit implementing the identiry matrix from two multi-controlled
    Pauli operator.

    Args:
        pauli_string (str, optional): The Pauli operator to approximate (default is 'x').
        error (float, optional): The error rate for the approximation (default is 0.1).

    Returns:
        QuantumCircuit: A quantum circuit implementing the approximate the identidy
        matrix from the multi-controlled single qubit gates.

    The function constructs a quantum circuit that approximates the unitary transformation
    corresponding to the specified Pauli operator. The approximation is performed with a
    given error rate. The circuit includes a set of control qubits, a target qubit, and a
    classical register for measurement outcomes.

    Example:
        >>> my_circuit = built_circuit('x', 0.1)
        >>> print(my_circuit)

                ┌───┐┌──────────────┐
    controls_0: ┤ X ├┤0             ├──■─────
                ├───┤│              │  │
    controls_1: ┤ X ├┤1             ├──■─────
                ├───┤│              │  │
    controls_2: ┤ X ├┤2             ├──■─────
                ├───┤│              │  │
    controls_3: ┤ X ├┤3 Ldmcuapprox ├──■─────
                ├───┤│              │  │
    controls_4: ┤ X ├┤4             ├──■─────
                ├───┤│              │  │
    controls_5: ┤ X ├┤5             ├──■─────
                ├───┤│              │┌─┴─┐┌─┐
        target: ┤ X ├┤6             ├┤ X ├┤M├
                └───┘└──────────────┘└───┘└╥┘
     classic: 1/═══════════════════════════╩═
                                           0
    """

    pauli_matrix = Ut.pauli_matrices(pauli_string)
    pauli_matrix_dagger = Ut.transpose_conjugate(pauli_matrix)
    n_base = Ut.pyramid_size(pauli_matrix_dagger, error)

    controls = QuantumRegister(n_base, 'controls')
    target = QuantumRegister(1, 'target')
    classical = ClassicalRegister(1, "classic")

    circ = QuantumCircuit(controls, target, classical)
    for i in range(len(controls)):
        circ.x(i)
    circ.x(target)
    MCU.mcu(circ, pauli_matrix_dagger, controls, target, error)
    circ.mcx(controls, target)
    circ.measure(target, [0])
    return circ


def counts_from_fake_backend(circ):
    """
    Simulates a quantum circuit with a basic device noise model using the `FakeVigo` backend.

    Args:
        circ (QuantumCircuit): The quantum circuit to be simulated.

    Returns:
        dict: A dictionary containing the measurement outcomes and their respective counts.

    Raises:
        QiskitError: If there is an issue with the quantum circuit or the simulation process.

    This function uses the `FakeVigo` backend from Qiskit's Aer simulator to model the noise
    in a quantum device. It builds a noise model based on the properties of the fake backend,
    including the coupling map and basis gates. The provided quantum circuit is then transpiled
    to match the backend's configuration, and a noise simulation is performed. The function
    returns a dictionary containing the measurement outcomes and their counts.

    Example:
        >>> from qiskit import QuantumCircuit
        >>> circ_example = QuantumCircuit(2, 2)
        >>> circ_example.h(0)
        >>> circ_example.cx(0, 1)
        >>> result_example = accuracy_from_fake_backend(circ)
        >>> print(result)
        {'00': 500, '11': 500}
    """

    backend = FakeVigo()
    noise_model = NoiseModel.from_backend(backend)

    coupling_map = backend.configuration().coupling_map

    basis_gates = noise_model.basis_gates

    backend = AerSimulator(noise_model=noise_model,
                           coupling_map=coupling_map,
                           basis_gates=basis_gates)
    transpiled_circuit = transpile(circ, backend)  # optimization_level=3)
    result = backend.run(transpiled_circuit).result()

    counts = result.get_counts(0)
    return counts


def vis(circ, counts):
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting the histogram in the first subplot
    plot_histogram(counts, ax=ax1)
    ax1.set_title('Histogram')

    # Drawing the circuit in the second subplot
    circuit_drawer(circ, ax=ax2, output='mpl')
    ax2.set_title('Circuit')

    # Fit layout
    plt.tight_layout()
    plt.show()


def pyramid_size_vs_error(operator, start, stop, n_samples):
    interval = np.linspace(start, stop, n_samples)
    dic = {}
    n_base = 0
    for error in interval:
        old_n_base = n_base
        n_base = Ut.pyramid_size(operator, error)
        if old_n_base != n_base:
            dic[error] = n_base
    return dic


def axis_to_plot(operator, start=1e-3):
    x_axis = []
    y_axis = []
    z_axis = []
    d = pyramid_size_vs_error(operator, start, 0.9, 100)
    for error in d:
        circuit = built_circuit('x', error)
        counts = counts_from_fake_backend(circuit)
        accuracy = counts['1'] / (counts['0'] + counts['1'])
        x_axis.append(error)
        y_axis.append(accuracy)
        z_axis.append(len(circuit))

    return np.array(x_axis), np.array(y_axis), np.array(z_axis)


def generate_accuracy(pauli_string):
    sqg = Ut.pauli_matrices(pauli_string)
    _, accuracy, _ = axis_to_plot(sqg, start=1e-1)
    return accuracy


def avg_accuracy(pauli_string, n_mean):
    # Paraleliza a geração de conjuntos de valores de accuracy
    accuracy_arr = Parallel(n_jobs=-1)(
        delayed(generate_accuracy)(pauli_string) for _ in range(n_mean))

    # Converte a lista de arrays em uma matriz NumPy
    accuracy_arr = np.array(accuracy_arr)

    # Adiciona um cabeçalho ao arquivo
    header = "Accuracy Data for Pauli String: {}".format(pauli_string)

    # Salva os dados no arquivo
    file_path = 'avg_accuracy.txt'
    np.savetxt(file_path, accuracy_arr, fmt='%.6f', header=header)


#
# beginning = time.time()
#
# avg_accuracy('x', 300)
#
# end = time.time()
# time_of_execution = end - beginning
# print(f"Time of execution: {time_of_execution} seconds")

single_qubit_gate = Ut.pauli_matrices('x')
beginning = time.time()
err, acc, len_circ = axis_to_plot(single_qubit_gate, start=1e-1)
end = time.time()
time_of_execution = end - beginning
print(f"Time of execution: {time_of_execution} seconds")

table = np.empty([err.size, 3])
table[:, 0] = err
table[:, 1] = acc
table[:, 2] = len_circ
file = 'error_accuracy_circuit-size.txt'
np.savetxt(file, table)

fig = plt.figure(figsize=(16, 12))

# 3D graph
ax_3d = fig.add_subplot(221, projection='3d')
ax_3d.scatter(err, len_circ, acc, color="green")
ax_3d.set_title('Error x size_circ x accuracy')
ax_3d.set_xlabel('Error')
ax_3d.set_ylabel('Size of circuit')
ax_3d.set_zlabel('Accuracy')

# 2D graph (err, acc)
ax_2d_1 = fig.add_subplot(222)
ax_2d_1.plot(err, acc, 'r-o')
ax_2d_1.set_title('Error x accuracy')
ax_2d_1.set_xlabel('Error')
ax_2d_1.set_ylabel('Accuracy')

# 2D graph (err, len_circ)
ax_2d_2 = fig.add_subplot(223)
ax_2d_2.plot(err, len_circ, 'b-o')
ax_2d_2.set_title('Error x size_circ')
ax_2d_2.set_xlabel('Error')
ax_2d_2.set_ylabel('Size_circ')

# 2D graph (len_circ, acc)
ax_2d_3 = fig.add_subplot(224)
ax_2d_3.plot(len_circ, acc, 'k-o')
ax_2d_3.set_title('size_circ x accuracy')
ax_2d_3.set_xlabel('size_circ')
ax_2d_3.set_ylabel('accuracy')

plt.savefig('result.png', dpi=300)
plt.tight_layout()
plt.show()
