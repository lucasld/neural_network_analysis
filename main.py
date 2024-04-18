import jax
import optax
import flax.linen.initializers as initializers
import flax.linen as nn
import os
import argparse

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


configurations = {
    'cifar10': {
        'xavier_uniform': {
            'tanh': (initializers.xavier_uniform(), nn.tanh),
            'relu': (initializers.xavier_uniform(), nn.relu),
            'sigmoid': (initializers.xavier_uniform(), nn.sigmoid)
        },
        'xavier_normal': {
            'tanh': (initializers.kaiming_normal(), nn.tanh),
            'relu': (initializers.kaiming_normal(), nn.relu),
            'sigmoid': (initializers.kaiming_normal(), nn.sigmoid)
        },
        'kaiming_uniform': {
            'tanh': (initializers.kaiming_uniform(), nn.tanh),
            'relu': (initializers.kaiming_uniform(), nn.relu),
            'sigmoid': (initializers.kaiming_uniform(), nn.sigmoid)
        },
        'kaiming_normal': {
            'tanh': (initializers.kaiming_normal(), nn.tanh),
            'relu': (initializers.kaiming_normal(), nn.relu),
            'sigmoid': (initializers.kaiming_normal(), nn.sigmoid)
        }
    },
    'wine_quality': {
        'xavier_uniform': {
            'tanh': (initializers.xavier_uniform(), nn.tanh),
            'relu': (initializers.xavier_uniform(), nn.relu),
            'sigmoid': (initializers.xavier_uniform(), nn.sigmoid)
        },
        'xavier_normal': {
            'tanh': (initializers.kaiming_normal(), nn.tanh),
            'relu': (initializers.kaiming_normal(), nn.relu),
            'sigmoid': (initializers.kaiming_normal(), nn.sigmoid)
        },
        'kaiming_uniform': {
            'tanh': (initializers.kaiming_uniform(), nn.tanh),
            'relu': (initializers.kaiming_uniform(), nn.relu),
            'sigmoid': (initializers.kaiming_uniform(), nn.sigmoid)
        },
        'kaiming_normal': {
            'tanh': (initializers.kaiming_normal(), nn.tanh),
            'relu': (initializers.kaiming_normal(), nn.relu),
            'sigmoid': (initializers.kaiming_normal(), nn.sigmoid)
        }
    }
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="one of [cifar10, wine_quality]",
                    type=str)
    parser.add_argument("init_method", help="one of [xavier_uniform, kaiming_normal,kaiming_uniform]",
                    type=str)
    parser.add_argument("activation_func", help="one of [tanh, relu,sigmoid]",
                    type=str)
    args = parser.parse_args()


    task = args.task
    init_method_str = args.init_method
    activation_func_str = args.activation_func
    
    # define a random number generator key (seed)
    rng = jax.random.PRNGKey(0)
    # create checkpoint path
    checkpoint_path = os.path.join(os.getcwd(), "model_checkpoints/",
                                   task, init_method_str, activation_func_str)
    # get respective functions from configuration-dict
    funcs = configurations[task][init_method_str][activation_func_str]
    init_method, activation_func = funcs
    
    if task == 'cifar10':
        training_cifar10(rng, init_method, activation_func, checkpoint_path)
    elif task == 'wine_quality':
        training_wine_quality(rng, init_method, activation_func, checkpoint_path)