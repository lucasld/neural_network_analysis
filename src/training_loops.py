import jax
from functools import partial
import optax
import flax.linen as nn
import jax.numpy as jnp

from models import create_model_cifar10
from data_loaders import load_cifar10


@jax.jit
def loss_fn_cnn10(logits, target_data):

    # Compute softmax cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits,target_data)
    return jnp.mean(loss)

@partial(jax.jit,static_argnums = 1)
@partial(jax.value_and_grad, argnums=0)
def forward_cnn10(weights, model, input_data, target_data):

    prediction = model.apply(weights,input_data)
    return loss_fn_cnn10(prediction,target_data)

@partial(jax.jit,static_argnums = [1,5])
def train_step_cnn10(weights, model, input_data, target_data, opt_state, optimizer):
    
    loss_v,grads = forward_cnn10(weights, model, input_data,target_data)

    updates, opt_state = optimizer.update(grads, opt_state)
    weights = optax.apply_updates(weights, updates)

    return loss_v,weights,opt_state


def train_loop(training_data, model, model_weights, optimizer, opt_state):
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Iterate over batches of training data
        for batch in training_data:
            input_data, target_data = batch
            print("type: " ,type(input_data))
            input_data = jnp.array(input_data)
            target_data = jnp.array(target_data)

            # Perform a training step
            loss, model_weights, opt_state = train_step_cnn10(model_weights, model, input_data, target_data, opt_state, optimizer)
            # Print or log the loss for monitoring
            print(f"Epoch {epoch + 1}, Loss: {loss}")



if __name__ == "__main__":
    # Load data
    train_dataset, validation_dataset, test_dataset = load_cifar10()
    # Define a random number generator key (seed)
    rng = jax.random.PRNGKey(0)
    # Initialize model, optimizer, and other necessary components
    model, weights = create_model_cifar10(rng)
    optimizer = optax.sgd(learning_rate=0.001)
    opt_state = optimizer.init(model)
    # train model
    train_loop(train_dataset, model, weights, optimizer, opt_state)


