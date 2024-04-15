import jax
import optax

from src.models import create_model_cifar10
from src.data_loaders import load_cifar10
from src.training_loops import train_loop


if __name__ == "__main__":
    # Load data
    train_dataset, validation_dataset, test_dataset = load_cifar10()
    # Define a random number generator key (seed)
    rng = jax.random.PRNGKey(0)
    # Initialize model, optimizer, and other necessary components
    model, weights = create_model_cifar10(rng)
    #optimizer = optax.adam(learning_rate=0.001)  # TODO: doesnt work yet
    optimizer = optax.sgd(learning_rate=0.001)
    opt_state = optimizer.init(model)
    # train model
    train_loop(train_dataset, validation_dataset, model, weights, optimizer, opt_state, epochs=5)


