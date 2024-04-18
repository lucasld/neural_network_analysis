import jax
import optax
import flax.linen.initializers as initializers
import flax.linen as nn

from src.models import Cifar10CNN, WineQualityNetwork, create_model
from src.data_loaders import load_cifar10, load_wine_quality
from src.training import train_loop, loss_fn_cnn10, loss_fn_wine
from src.utils import create_folder


def training_cifar10(rng, init_func, activation_func, checkpoint_path):
    create_folder(checkpoint_path)
    # load data
    train_dataset, validation_dataset, test_dataset = load_cifar10()
    # initialize model, optimizer, and other necessary components
    model, weights = create_model(Cifar10CNN, rng, init_func=init_func, activation_func=activation_func)
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(weights)
    # train model
    train_loop(train_dataset, validation_dataset, model, weights, optimizer,
               opt_state, loss_fn_cnn10, checkpoint_path, epochs=15,
               plot_every=1)
    

def training_wine_quality(rng, init_func, activation_func, checkpoint_path):
    create_folder(checkpoint_path)
    # load data
    train_dataset, validation_dataset, test_dataset = load_wine_quality()
    # initialize model, optimizer, and other necessary components
    model, weights = create_model(WineQualityNetwork, rng,
                                  input_shape=(1, 11), init_func=init_func,
                                  activation_func=activation_func)
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(weights)
    # train model
    train_loop(train_dataset, validation_dataset, model, weights, optimizer,
               opt_state, loss_fn_wine, checkpoint_path, epochs=600,
               plot_every=50)


if __name__ == "__main__":
    # define a random number generator key (seed)
    rng = jax.random.PRNGKey(0)
    checkpoint_path = f"/Users/lld/Documents/SoSe 2024/deep_neural_network_analysis/neural_network_analysis/model_checkpoints/cifar10/xavier/tanh"
    #checkpoint_path = f"/Users/lld/Documents/SoSe 2024/deep_neural_network_analysis/neural_network_analysis/model_checkpoints/wine_quality/xavier/tanh"
    #training_wine_quality(rng, 'xavier')
    training_cifar10(rng, initializers.xavier_uniform(), nn.tanh, checkpoint_path)

