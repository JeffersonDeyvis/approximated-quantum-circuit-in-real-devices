""""
"""
from typing import Union
from qclib.gates.mcu import MCU
import numpy as np


class Utilities:

    def __init__(self, operator, error):
        self.operator = operator
        self.error = error

    @staticmethod
    def pauli_matrices(string: str) -> Union[np.ndarray, None]:
        string_case = string.lower()
        match string_case:
            case "x":
                return np.array([
                    [0, 1],
                    [1, 0]
                ])
            case "y":
                return np.array([
                    [0, -1j],
                    [1j, 0]
                ])
            case "z":
                return np.array([
                    [1, 0],
                    [0, -1]
                ])
            case _:
                return None

    @staticmethod
    def transpose_conjugate(operator):
        return np.conjugate(np.transpose(operator))

    @staticmethod
    def pyramid_size(operator, error):
        mcu_dummy = MCU(operator, num_controls=100, error=error)
        # will be changed by mcu_dummy._get_n_base(x_dagger, error)
        return mcu_dummy._get_num_base_ctrl_qubits(operator, error)
