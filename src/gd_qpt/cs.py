# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of compressed sensing following QISKIT"""
import cvxpy

from scipy import sparse as sps

import numpy as np

from jax import jit, vmap
from jax import numpy as jnp


@jit
def vec(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorize, or "vec", a matrix by column stacking.
    For example the 2 by 2 matrix A::
        A = [[a, b]
             [c, d]]
    becomes::
      |A>> := vec(A) = (a, c, b, d)^T
    where `|A>>` denotes the vec'ed version of A and :math:`^T` denotes transpose.
    :param matrix: A N (rows) by M (columns) numpy array.
    :return: Returns a column vector with  N by M rows.
    """
    return matrix.reshape((-1, 1))# matrix.T.reshape((-1, 1))

def _row_sensing_matrix(probe, measurement):
    """Construct a row of the sensing matrix S.
    
    The probe and measurement are operators in a Hilbert space. In case of qubits
    the probes and measurements are tensor products of single qubit Pauli ops, 
    e.g., for a three qubit system where X, Y, Z are single-qubit Pauli operators,
    the probes and measurements can be XXX, XYZ, X_YZ. Where X, X_ are the
    eigenstates of the Pauli operator X. 

    The matrix is constructed using the description in Appx. A of Knee et. al,
    2019 (https://doi.org/10.1103/PhysRevA.98.062336).


    Args:
        probe (array): The probe operator.
        measurement (array): The measurement (tensor product )

    Returns:
        array: A row of the sensing matrix.
    """
    return vec(jnp.kron(probe, measurement)).T[0]


get_sensing_matrix = jit(vmap(vmap(_row_sensing_matrix, in_axes=[None, 0]), in_axes=[0, None]))


def partial_trace_super(dim1: int, dim2: int) -> np.array:
    """
    Return the partial trace superoperator in the column-major basis.
    This returns the superoperator S_TrB such that:
        S_TrB * vec(rho_AB) = vec(rho_A)
    for rho_AB = kron(rho_A, rho_B)
    Args:
        dim1: the dimension of the system not being traced
        dim2: the dimension of the system being traced over
    Returns:
        A Numpy array of the partial trace superoperator S_TrB.
    """

    iden = sps.identity(dim1)
    ptr = sps.csr_matrix((dim1 * dim1, dim1 * dim2 * dim1 * dim2))

    for j in range(dim2):
        v_j = sps.coo_matrix(([1], ([0], [j])), shape=(1, dim2))
        tmp = sps.kron(iden, v_j.tocsr())
        ptr += sps.kron(tmp, tmp)

    return ptr


class CompressedSensing(object):
    def __init__(self, dim:int):
        """Initialization of the the fitter

        Args:
            dim (int): Hilbert space dimension
        """
        rho_r = cvxpy.Variable((dim, dim), symmetric=True)
        rho_i = cvxpy.Variable((dim, dim))
        cons = [rho_i == -rho_i.T]
        rho = cvxpy.bmat([[rho_r, -rho_i], [rho_i, rho_r]])
        cons.append(rho >> 0)
        sdim = int(np.sqrt(dim))
        ptr = partial_trace_super(sdim, sdim)
        cons.append(ptr @ cvxpy.vec(rho_r) == np.identity(sdim).ravel())
        cons.append(ptr @ cvxpy.vec(rho_i) == np.zeros(sdim*sdim))

        self.dim = dim
        self.rho_r = rho_r
        self.rho_i = rho_i
        self.cons = cons
        self.sdim = sdim
        self.ptr = ptr


    def fit(self, ops_matrix, data_vector):
        """_summary_

        Args:
            ops_matrix (_type_): _description_
            data_vector (_type_): _description_
        """
        bm_r = np.real(ops_matrix)
        bm_i = np.imag(ops_matrix)

        arg = bm_r @ cvxpy.vec(self.rho_r) - bm_i @ cvxpy.vec(self.rho_i) - np.array(data_vector)

        # SDP objective function
        obj = cvxpy.Minimize(cvxpy.norm(arg, p=2))

        # Solve SDP
        prob = cvxpy.Problem(obj, self.cons)
        prob.solve(verbose=False, solver=cvxpy.SCS, eps=1e-10)
        rho_fit = self.rho_r.value + 1j * self.rho_i.value
        return rho_fit
