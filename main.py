import jax
import optax

from src.models import Cifar10CNN, WineQualityNetwork, create_model
from src.data_loaders import load_cifar10, load_wine_quality
from src.training_loops import train_loop, loss_fn_cnn10, loss_fn_wine


def training_cifar10():
    # load data
    train_dataset, validation_dataset, test_dataset = load_cifar10()
    # define a random number generator key (seed)
    rng = jax.random.PRNGKey(0)
    # initialize model, optimizer, and other necessary components
    model, weights = create_model(Cifar10CNN, rng, init_method='xavier')
    #optimizer = optax.adam(learning_rate=0.001)  # TODO: doesnt work yet
    optimizer = optax.sgd(learning_rate=0.001)
    opt_state = optimizer.init(model)
    checkpoint_path = "/Users/lld/Documents/SoSe 2024/deep_neural_network_analysis/neural_network_analysis/model_checkpoints/cifar10/xavier"
    # train model
    train_loop(train_dataset, validation_dataset, model, weights, optimizer,
               opt_state, loss_fn_cnn10, checkpoint_path, epochs=200)
    

def training_wine_quality():
    # load data
    train_dataset, validation_dataset, test_dataset = load_wine_quality()
    # define a random number generator key (seed)
    rng = jax.random.PRNGKey(0)
    # initialize model, optimizer, and other necessary components
    model, weights = create_model(WineQualityNetwork, rng,
                                  input_shape=(1, 11, 1), init_method='xavier')
    #optimizer = optax.adam(learning_rate=0.001)  # TODO: doesnt work yet
    optimizer = optax.sgd(learning_rate=0.001)
    opt_state = optimizer.init(model)
    checkpoint_path = "/Users/lld/Documents/SoSe 2024/deep_neural_network_analysis/neural_network_analysis/model_checkpoints/wine_quality/xavier"
    # train model
    train_loop(train_dataset, validation_dataset, model, weights, optimizer,
               opt_state, loss_fn_wine, checkpoint_path, epochs=40)


if __name__ == "__main__":
    training_wine_quality()


