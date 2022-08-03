"""Neural network quantum process tomography"""
import haiku as hk


import jax


class Encoder(hk.Module):
    """Encoder model that goes from data -> kraus_ops."""
    def __init__(self, hilbert_size, num_kraus=1):
        super().__init__()
        self.hilbert_size = hilbert_size
        self.num_kraus = num_kraus

    def __call__(self, x):
        x = hk.Flatten()(x)
        x = x.reshape(1, -1)

        x = hk.Linear(128)(x)
        x = jax.nn.tanh(x)

        x = hk.Linear(128)(x)
        x = jax.nn.tanh(x)

        x = hk.Linear(self.hilbert_size**2*self.num_kraus*2)(x)
        x = x.reshape(self.num_kraus, self.hilbert_size, self.hilbert_size, 2)
        
        k_ops = x.reshape(self.num_kraus, self.hilbert_size, self.hilbert_size, 2)
        
        return k_ops[..., 0] + 1j*k_ops[..., 1]
