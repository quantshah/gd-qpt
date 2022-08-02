"""Gradient-descent quantum process tomography"""
from functools import partial

import numpy as np


import jax
from jax import jit
from jax import vmap
import jax.numpy as jnp
from jax.config import config

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
    """Loss function for the trainnig.

    Args:
        params (_type_): Network parameters
        rng (_type_): Random number seq
        d_batch (_type_): A (batchsize x batchsize) input of probabilies
        probes (_type_): The probe operators. Note that these are assumed to be
                         Pauli vectors that the predict function knows how
                         to use.
        measurements (_type_): The measurement operators as Pauli vectors.

    Returns:
        loss (float): A scalar loss
    """
    k_ops = get_unblock(params, num_kraus)
    data_pred = predict(k_ops, probes, measurements)
    l2 = jnp.sum(((data - data_pred)**2))
    return l2 + 0.01*jnp.linalg.norm(params, ord=1)


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



class GradientDescent(object):
    def __init__(self, dim:int, num_kraus:int, lr=0.1, alpha=0.999,
                 batch_size=32, maxiter=10000):
        """Initialization of the the fitter.

        Args:
            dim (int): Hilbert space dimension
        """
        params_init = jnp.array([rand_unitary(2**k, density=0.5).full()/np.sqrt(num_kraus) for w in range(num_kraus)])
        params = get_block(params_init)

        self.dim = dim
        self.num_kraus = num_kraus
        self.params = params
        self.lr = lr 
        self.alpha = alpha
        self.params = params
        self.batch_size = batch_size
        self.maxiter = maxiter


    def step(self, A, B, data):
        """_summary_

        Args:
            ops (_type_): _description_
            data_vector (_type_): _description_
        """
        grads = jax.grad(loss)(self.params, data, A, B, num_kraus=self.num_kraus)
        grads = jnp.conj(grads)
        grads = grads/jnp.linalg.norm(grads)
        self.params = stiefel_update(self.params, grads, self.lr)
        self.lr = self.alpha*self.lr

    def fit(self, ops, data):
        """_summary_
        """
        k_ops = get_unblock(self.params, self.num_kraus)
        choi_gd_pred = choi(k_ops)

        return choi_gd_pred
