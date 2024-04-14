import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.common_utils import onehot
import flax

class Cifar10CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x

def create_model_cifar10(rng, input_shape=(32, 32, 3)):
    model = Cifar10CNN()
    variables = model.init(rng, jnp.ones(input_shape, jnp.float32))
    weights = flax.jax_utils.unreplicate(variables['params'])  # Extract weights
    return model, weights
