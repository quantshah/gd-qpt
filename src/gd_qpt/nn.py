"""Neural network quantum process tomography"""
import haiku as hk


import jax


class Encoder(hk.Module):
    """Encoder model that goes from data -> Kraus operators."""
    def __init__(self, N, num_kraus=1):
        """Initialization.

        Args:
            N (int): Hilbert space dimension.
            num_kraus (int, optional): Number of Kraus operators. Defaults to 1.
        """
        super().__init__()
        self.N = N
        self.num_kraus = num_kraus

    def __call__(self, x):
        """The model forward function that takes in some data, flattens it,
        passes it through the model and generates a set of Kraus operators.

        Args:
            x (array): An array of expectation values.

        Returns:
            K (array[complex]): A complex-valued array (kN, N) of a 
                                    block of Kraus ops for a batch of the 
                                    input data x.
        """
        x = hk.Flatten()(x)
        x = x.reshape(1, -1)

        x = hk.Linear(128)(x)
        x = jax.nn.tanh(x)

        x = hk.Linear(128)(x)
        x = jax.nn.tanh(x)

        x = hk.Linear(self.N**2*self.num_kraus*2)(x)
        x = x.reshape(self.num_kraus, self.N, self.N, 2)
        
        k_ops = x.reshape(self.num_kraus, self.N, self.N, 2)
        K = get_block(k_ops[..., 0] + 1j*k_ops[..., 1])
        return K
