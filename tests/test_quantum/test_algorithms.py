import numpy as np

from qconduit.quantum import algorithms, simulation


def constant_oracle(_: int) -> int:
    return 0


def balanced_oracle(x: int) -> int:
    return bin(x).count("1") % 2


def test_deutsch_jozsa_constant_vs_balanced():
    constant_result = algorithms.deutsch_jozsa(constant_oracle, n_qubits=3)
    balanced_result = algorithms.deutsch_jozsa(balanced_oracle, n_qubits=3)
    assert constant_result == "000"
    assert balanced_result != "000"


def simon_oracle_factory(secret: int, n_qubits: int):
    mask = (1 << n_qubits) - 1

    def oracle(x: int) -> int:
        partner = x ^ secret
        return min(x & mask, partner & mask)

    return oracle


def test_simon_two_and_three_qubits():
    for n, secret in [(2, 0b11), (3, 0b101)]:
        oracle = simon_oracle_factory(secret, n)
        result = algorithms.simon(oracle, n)
        assert result == format(secret, f"0{n}b")


def test_grover_single_and_multiple_solutions():
    single_solution = algorithms.grover(lambda x: int(x == 5), n_qubits=3)
    assert single_solution == 5
    multi_result = algorithms.grover(lambda x: int(x in {1, 6}), n_qubits=3)
    assert multi_result in {1, 6}


def test_qft_and_inverse_round_trip():
    n = 3
    state = simulation.basis_state(1, n)
    transformed = algorithms.qft(state)
    matrix_transform = algorithms.qft_matrix(n) @ state
    assert np.allclose(transformed, matrix_transform)
    recovered = algorithms.iqft(transformed)
    assert np.allclose(recovered, state)
    unitary = algorithms.qft_matrix(n) @ algorithms.iqft_matrix(n)
    assert np.allclose(unitary, np.eye(1 << n))


