"""Gradient-descent quantum process tomography"""
from functools import partial

import numpy as np


from qutip import rand_unitary

import jax
from jax import jit
from jax import vmap
import jax.numpy as jnp
from jax.config import config

from tqdm.auto import tqdm

from gd_qpt.core import choi


config.update("jax_enable_x64", True)


def _predict(channel, op1,  op2):
    """
    Predicts the probabilities for Pauli measurements. 

    For a given Pauli input state and measurement basis, sums to
    one. Hence total sum is $18^k$.

    Output array $(6^k, 6^k)$. First coordinate input state, second 
    coordinate measured output.
    """
    res = 0.
    
    # Looping over kraus instead of doing everything in the einsum to
    # avoid excessive memory usage if the rank is high.
    for kraus in channel:
        # a = jnp.einsum("nj, ij, mi -> nm", op1, kraus, dag(op2), optimize="optimal")
        a = op1.T.conj()@kraus@op2
        res += (a.real**2 + a.imag**2) 
    return res

predict = vmap(vmap(_predict, in_axes=[None, 0, None]), in_axes=[None, None, 0])

@partial(jit, static_argnums=4)
def loss(params, data=None, probes=None, measurements=None, num_kraus=None):
    """Loss function for the training assuming a predict function that can 
    generate probabilities for a measurement from the given process representation
    captured in params.

    Args:
        params (array): Parameters to optimize, e.g., Kraus operators.
        data (array): Data representing measured probabilities.
        probes (array): The probe operators.
        measurements (array): The measurement operators as Pauli vectors.
        num_kraus (int): The number of Kraus operators.

    Returns:
        loss (float): A scalar loss
    """
    k_ops = get_unblock(params, num_kraus)
    data_pred = predict(k_ops, probes, measurements)

    l2 = jnp.sum(((data - data_pred)**2))
    return l2 + 0.001*jnp.linalg.norm(params, 1)


@jit
def stiefel_update(params,
                   grads,
                   step_size):
    """Updates params in the direction of gradients while staying on the
    Stiefel manifold

    Args:
        params (array[complex]): (n x m) array representing parameters to update 
        grads (array[complex]): (n x m) array representing gradients (note)
        step_size (float): The step size for the update

    Returns:
        updated_params (array[complex]): Updated parameters
    """
    U = jnp.hstack([grads, params])
    V = jnp.hstack([params, -grads])

    prod_term = V.T.conj()@U
    invterm = jnp.eye(prod_term.shape[0]) + (step_size/2.)*prod_term
    A = step_size*(U@jnp.linalg.inv(invterm))

    B = V.T.conj()@params

    updated_params = params - A@B
    return updated_params


@jit
def get_block(kops):
    """Get a block matrix from the operators

    Args:
        kops (array[compelx]): A (N, k, k) array of Kraus operators
    """
    return jnp.concatenate([*kops])


@partial(jit, static_argnums=1)
def get_unblock(kmat, num_kraus):
    """Get Kraus operators from a block form

    Args:
        kmat (array[compelx]): A (Nk, k) matrix
    """
    return jnp.array(jnp.vsplit(kmat, num_kraus))



def generate_batch(batch_size, num_probes, num_measurements):
    """Generates random indices to select a batch of the data 
        (probes x measurements) assuming same number of probes and measurements

    Args:
        batch_size (int): Batch size
        len_indices (int): Length of training data 
                          (probes and measurements are assumed to be the same)

    Returns:
        idx : A meshgrid of indices for selecting the data.
        idx1, idx2 (array): Indices for the probes and measurements.
    """
    idx1 = np.random.randint(0, num_probes, size=[batch_size])
    idx2 = np.random.randint(0, num_measurements, size=[batch_size])
    idx = tuple(np.meshgrid(idx1, idx2))
    return idx, idx1, idx2

class GradientDescent(object):
    def __init__(self, N:int, num_kraus:int, lr=0.1, alpha=0.999):
        """Initialization of the the fitter.

        Args:
            N (int): Hilbert space dimension
        """
        self.N = N
        self.num_kraus = num_kraus
        self.lr = lr 
        self.alpha = alpha

    def fit(self, data, probes, measurements, batch_size, maxiters=1000):
        """_summary_
        """
        params_init = jnp.array([rand_unitary(self.N,
                                 density=0.5).full()/np.sqrt(self.num_kraus) for w in range(self.num_kraus)])
        params = get_block(params_init)
        lr = self.lr

        for step in tqdm(range(maxiters)):
            idx, idx1, idx2 = generate_batch(batch_size, probes.shape[0], measurements.shape[0])

            grads = jax.grad(loss)(params, data.T[idx].real, probes[idx1], measurements[idx2],
                                   num_kraus=self.num_kraus)
            grads = jnp.conj(grads)
            grads = grads/jnp.linalg.norm(grads)

            params = stiefel_update(params, grads, lr)
            lr = self.alpha*lr

        k_ops = get_unblock(params, self.num_kraus)
        choi_gd_pred = choi(k_ops)

        return choi_gd_pred
