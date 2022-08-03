"""Core utility functions used throughout the code"""
from itertools import product

import numpy as np

from qutip import tensor

from jax import numpy as jnp
from jax import jit
from jax.config import config

config.update("jax_enable_x64", True)


def tensor_product_list(arr: list, repeat: int)->list:
    """Create a list with all tensor products of elements in arr.

    Uses the itertools.product function to construct all possible permutations.

    Args:
        arr (list): A list of qutip.Qobj representing states or operators.
        repeat (int): The number of elements to permute. 

    Returns:
        prod (list): Tensor products for all combination of elements in arr.
    """
    prod = []
    for p in product(arr, repeat=repeat):
        prod.append(tensor(*p))
    return prod


def convert_to_jax(arr: list)->jnp.array:
    """Converts QuTiP arrays to Jax arrays.

    Args:
        arr (list): A list of qutip objects to convert.

    Returns:
        jnp.array: Jax version of input arrays
    """
    return jnp.array([op.full() for op in arr])


@jit
def dag(op: jnp.array)->jnp.array:
    """Dagger operation on an operator

    Args:
        op (jnp.array): Any operator to take the dagger operation on.

    Returns:
        jnp.array: Conjuage transpose of the operator.
    """
    return jnp.conjugate(jnp.transpose(op))


def prod_pauli_vecs(k, U2=None):
    """Outputs all the k-tensor products of Pauli vectors, as an array where the
    vectors are the lines.

    Original code from https://github.com/Hannoskaj/Hyperplane_Intersection_Projection

    Note:
    This implementation works till k=8. It needs $12^k$ complex entries.
    U2 allows to add a rotation to the Pauli vectors to avoid very special cases.

    TODO: There could be faster and more general way to do this.

    Args:
        k (int): The number of qubits.
        U2 (np.array, optional): A unitary of dimension 2**k. Defaults to None.

    Returns:
        [type]: [description]
    """
    s2 = np.sqrt(0.5)
    frame_vecs = np.array(
        ([1, 0], [0, 1], [s2, s2], [s2, -s2], [s2, s2 * 1j], [s2, -s2 * 1j])
    )
    if U2 is not None:
        frame_vecs = np.dot(frame_vecs, U2)
    einstein_indices = (
        "ai -> ai",
        "ai, bj -> abij",
        "ai, bj, ck -> abcijk",
        "ai, bj, ck, dl -> abcdijkl",
        "ai, bj, ck, dl, em -> abcdeijklm",
        "ai, bj, ck, dl, em, fn -> abcdefijklmn",
        "ai, bj, ck, dl, em, fn, go -> abcdefgijklmno",
        "ai, bj, ck, dl, em, fn, go, hp -> abcdefghijklmnop",
    )
    return np.einsum(einstein_indices[k - 1], *([frame_vecs] * k)).reshape(6 ** k, -1)


def probas_pauli(k, channel, optimize="optimal"):
    """Yields the probability of each Pauli measurement result for direct
    measurement method.

    Original code from https://github.com/Hannoskaj/Hyperplane_Intersection_Projection

    For a given Pauli input state and measurement basis, sums to
    one. Hence total sum is $18^k$.

    Input: k is the number of qubits,
           channel are the Kraus operators of the channel.
    Output array $(6^k, 6^k)$. First coordinate input state, second 
    coordinate measured output.

    Args:
        k (int): Number of qubits
        channel (np.ndarray): The channel represented by an (r, d, d) dim array
        optimize (str, optional): Einsum optimization strategy.
                                  Defaults to "optimal".

    Returns:
        res (np.ndarray): The probabilities for all combinations of Pauli input
                          states and measurements as a (6**k, 6**k) matrix.
    """
    res = 0
    Pk = prod_pauli_vecs(k)
    # Looping over kraus instead of doing everything in the einsum to
    # avoid excessive memory usage if the rank is high.
    for kraus in channel:
        a = np.einsum("nj, ij, mi -> nm", Pk, kraus, Pk.conj(), optimize="optimal")
        res += a.real ** 2 + a.imag ** 2
    return res


def M_k(k):
    """Yields least-square estimators components for the input (or the output)
    state.

    Original code  from https://github.com/Hannoskaj/Hyperplane_Intersection_Projection

    Args:
        k (int): The number of qubits.

    Returns:
        mkmat (np.ndarray) : A (6^k, 2^k, 2^k) matrix. The first index
                             represents the index of the input state. The rest
                             of the indices represent the corresponding matrix.
    """
    P1 = prod_pauli_vecs(1)
    # ALERT (original comment by Jonas Kahn)
    #
    # Here I do not understand the position of the conj(). I would have thought
    # it is on the other P1.
    # But it is this way that yields the right result.
    M_1 = np.einsum("nd, ne -> nde", 3 * P1, P1.conj()) - np.eye(2)
    mkmat = np.copy(M_1)

    for i in range(2, k + 1):
        mkmat = np.einsum("nde, mfg -> nmdfeg", mkmat, M_1)
        mkmat = mkmat.reshape(6 ** i, 2 ** i, 2 ** i)
    return mkmat


def choi(kraus_ops):
    """Takes the Kraus reprensentation of a channel and returns the Choi matrix.

    Args:
        kraus_ops (np.ndarray): The (k, N, N)-array representing k Kraus ops.

    Returns:
        np.array: A (N^2, N^2) array representing the Choi matrix.
    """
    r, N, N = kraus_ops.shape
    vectorized_kraus = kraus_ops.reshape(r, N ** 2)
    return np.einsum("ij, il -> jl", vectorized_kraus, vectorized_kraus.conj())
