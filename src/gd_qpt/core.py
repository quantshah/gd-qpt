"""Core utility functions used throughout the code"""
from itertools import product

import numpy as np

from qutip import tensor

from jax import numpy as jnp
from jax import jit
from jax.config import config

config.update("jax_enable_x64", True)


import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib


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


def destroy(N):
    """Destruction (lowering or annihilation) operator.
    
    Args:
        N (int): Dimension of Hilbert space.
    Returns:
         :obj:`jnp.ndarray`: Matrix representation for an N-dimensional annihilation operator
    """
    if not isinstance(N, (int, jnp.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be an integer value")
    data = jnp.sqrt(jnp.arange(1, N, dtype=jnp.float32))
    mat = np.zeros((N, N))
    np.fill_diagonal(
        mat[:, 1:], data
    )  # np.full_diagonal is not implemented in jax.numpy
    return jnp.asarray(mat, dtype=jnp.complex64)  # wrap as a jax.numpy array


# TODO: apply jax device array data type to everything all at once
# ind = jnp.arange(1, N, dtype=jnp.float32)
# ptr = jnp.arange(N + 1, dtype=jnp.float32)
# ptr = index_update(
#    ptr, index[-1], N - 1
# )    index_update mutates the jnp array in-place like numpy
# return (
#    csr_matrix((data, ind, ptr), shape=(N, N))
#    if full is True
#    else csr_matrix((data, ind, ptr), shape=(N, N)).toarray()
# )


def create(N):
    """Creation (raising) operator.
    Args:
        N (int): Dimension of Hilbert space 
    Returns:
         :obj:`jnp.ndarray`: Matrix representation for an N-dimensional creation operator
    """
    if not isinstance(N, (int, jnp.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be an integer value")
    data = jnp.sqrt(jnp.arange(1, N, dtype=jnp.float32))
    mat = np.zeros((N, N))
    np.fill_diagonal(mat[1:], data)  # np.full_diagonal is not implemented in jax.numpy
    return jnp.asarray(mat, dtype=jnp.complex64)  # wrap as a jax.numpy array
    # ind = jnp.arange(0, N - 1, dtype=jnp.float32)
    # ptr = jnp.arange(N + 1, dtype=jnp.float32)
    # ptr = index_update(
    #    ptr, index[0], 0
    # )  # index_update mutates the jnp array in-place like numpy
    # return (
    #    csr_matrix((data, ind, ptr), shape=(N, N))
    #    if full is True
    #    else csr_matrix((data, ind, ptr), shape=(N, N)).toarray()
    # )
    # return data


def _kth_diag_indices(a, k):
    rows, cols = jnp.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def destroy(N):
    """Destruction (lowering or annihilation) operator.
    
    Args:
        N (int): Dimension of Hilbert space.
    Returns:
         :obj:`jnp.ndarray`: Matrix representation for an N-dimensional annihilation operator
    """
    if not isinstance(N, (int, jnp.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be an integer value")
    data = jnp.sqrt(jnp.arange(1, N, dtype=jnp.float32))
    mat = np.zeros((N, N))
    np.fill_diagonal(
        mat[:, 1:], data
    )  # np.full_diagonal is not implemented in jax.numpy
    return jnp.asarray(mat, dtype=jnp.complex64)  # wrap as a jax.numpy array


# TODO: apply jax device array data type to everything all at once
# ind = jnp.arange(1, N, dtype=jnp.float32)
# ptr = jnp.arange(N + 1, dtype=jnp.float32)
# ptr = index_update(
#    ptr, index[-1], N - 1
# )    index_update mutates the jnp array in-place like numpy
# return (
#    csr_matrix((data, ind, ptr), shape=(N, N))
#    if full is True
#    else csr_matrix((data, ind, ptr), shape=(N, N)).toarray()
# )


def create(N):
    """Creation (raising) operator.
    Args:
        N (int): Dimension of Hilbert space 
    Returns:
         :obj:`jnp.ndarray`: Matrix representation for an N-dimensional creation operator
    """
    if not isinstance(N, (int, jnp.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be an integer value")
    data = jnp.sqrt(jnp.arange(1, N, dtype=jnp.float32))
    mat = np.zeros((N, N))
    np.fill_diagonal(mat[1:], data)  # np.full_diagonal is not implemented in jax.numpy
    return jnp.asarray(mat, dtype=jnp.complex64)  # wrap as a jax.numpy array
    # ind = jnp.arange(0, N - 1, dtype=jnp.float32)
    # ptr = jnp.arange(N + 1, dtype=jnp.float32)
    # ptr = index_update(
    #    ptr, index[0], 0
    # )  # index_update mutates the jnp array in-place like numpy
    # return (
    #    csr_matrix((data, ind, ptr), shape=(N, N))
    #    if full is True
    #    else csr_matrix((data, ind, ptr), shape=(N, N)).toarray()
    # )
    # return data


def _kth_diag_indices(a, k):
    rows, cols = jnp.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


class Displace:
    r"""Displacement operator for optical phase space.
    
    .. math:: D(\alpha) = \exp(\alpha a^\dagger -\alpha^* a)
    Args:
    n (int): dimension of the displace operator
    """

    def __init__(self, n):
        # The off-diagonal of the real-symmetric similar matrix T.
        sym = (2.0 * (jnp.arange(1, n) % 2) - 1) * jnp.sqrt(jnp.arange(1, n))
        # Solve the eigensystem.
        mat = jnp.zeros((n, n), dtype=jnp.complex128)

        i, j = _kth_diag_indices(mat, -1)
        mat = mat.at[i, j].set(sym)

        i, j = _kth_diag_indices(mat, 1)
        mat = mat.at[i, j].set(sym)

        self.evals, self.evecs = jnp.linalg.eigh(mat)
        self.range = jnp.arange(n)
        self.t_scale = 1j ** (self.range % 2)

    def __call__(self, alpha):
        r"""Callable with ``alpha`` as the displacement parameter
        Args:
            alpha (float): Displacement parameter
        Returns:
            :obj:`jnp.ndarray`: Matrix representing :math:`n-`dimensional displace operator
            with :math:`\alpha` displacement
        
        """
        # Diagonal of the transformation matrix P, and apply to eigenvectors.
        transform = jnp.where(
            alpha == 0,
            self.t_scale,
            self.t_scale * (alpha / jnp.abs(alpha)) ** -self.range,
        )
        evecs = transform[:, None] * self.evecs
        # Get the exponentiated diagonal.
        diag = jnp.exp(1j * jnp.abs(alpha) * self.evals)
        return jnp.conj(evecs) @ (diag[:, None] * evecs.T)


def dag(state):
    r"""Returns conjugate transpose of a given state, represented by :math:`A^{\dagger}`, where :math:`A` is
    a quantum state represented by a ket, a bra or, more generally, a density matrix.
    Args:
        state (:obj:`jnp.ndarray`): State to perform the dagger operation on
     
    Returns:
        :obj:`jnp.ndarray`: Conjugate transposed jax.numpy representation of the input state
 
    """
    return jnp.conjugate(jnp.transpose(state))

@jit
def expect(oper, state):
    """Calculate the expectation value of an operator with respect to a density matrix
    
    Args:
        oper, state (ndarray): The operator and state of dimensions (N, N) where N is the
                               Hilbert space size
    """
    # convert to jax.numpy arrays in case user gives raw numpy
    oper, state = jnp.asarray(oper), jnp.asarray(state)
    # Tr(rho*op)
    return jnp.trace(jnp.dot(state, oper)).real


def plot_choi(choi, title="", y=0.95, norm=None, cbar=False):
    """Plot the real and imaginary parts of the Choi matrix.

    Args:
        choi (np.array): The Choi matrix for the process
        title (str, optional): The title for the plot.
        norm (colors.TwoSlopeNorm, optional): The normalization for the plot.
    """
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    cmap = "RdBu"

    
    norm = colors.TwoSlopeNorm(vmin=-np.max(np.abs(choi.real)), vcenter=0, vmax=np.max(np.abs(choi.real)))
    im = ax[0].matshow(choi.real, cmap=cmap, norm=norm)
    norm = colors.TwoSlopeNorm(vmin=-np.max(np.abs(choi.imag)), vcenter=0, vmax=np.max(np.abs(choi.imag)))
    im2 = ax[1].matshow(choi.imag, cmap=cmap, norm=norm)

    if cbar is True:
        cbar_ax = plt.colorbar(im, ax=[ax[0]], fraction=0.021, pad=0.04)
        cbar_ax = plt.colorbar(im2, ax=[ax[1]], fraction=0.021, pad=0.04)
        # ticks=[-1, -0.5, 0, 0.5, 1]
        # cbar_ax.ax.set_yticklabels(["-1", "-0.5", "0", "0.5", "1"])

    ax[0].set_xlabel("Re")
    ax[1].set_xlabel("Im")

    ax[0].set_xticks([0, int(choi.shape[0]/2), choi.shape[0]])
    ax[0].set_yticks([0, int(choi.shape[0]/2), choi.shape[0]])

    ax[0].set_xticks([0, int(choi.shape[0]/2), choi.shape[0]])
    ax[0].set_yticks([0, int(choi.shape[0]/2), choi.shape[0]])

    plt.suptitle(title, y=y)
