import jax
from functools import partial
import optax
import flax.linen as nn
import jax.numpy as jnp
import orbax.checkpoint
from flax.training import checkpoints
import os
import shutil
import matplotlib.pyplot as plt


# TODO: check if model saving is enough like this by loading it in
# TODO: add comments


@jax.jit
def loss_fn_cnn10(logits, target_data):
    # Compute softmax cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits,target_data)
    return jnp.mean(loss)


@partial(jax.jit, static_argnums=1)
@partial(jax.value_and_grad, argnums=0)
def forward_cnn10(weights, model, input_data, target_data):
    prediction = model.apply(weights,input_data)
    return loss_fn_cnn10(prediction,target_data)


@partial(jax.jit, static_argnums=[1,5])
def train_step_cnn10(weights, model, input_data, target_data, opt_state, optimizer):
    loss_v,grads = forward_cnn10(weights, model, input_data,target_data)

    updates, opt_state = optimizer.update(grads, opt_state)
    weights = optax.apply_updates(weights, updates)

    return loss_v,weights,opt_state


def train_loop(training_data, validation_data, model, model_weights, optimizer, opt_state, epochs=10):
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    val_loss_acc = []
    # save initial weights
    save_weights(model_weights, checkpointer, "initial_weights")
    best_loss = float('inf')

    # training
    for epoch in range(epochs):
        # iterate over batches of training data
        for batch in training_data:
            input_data, target_data = batch
            print("type: " ,type(input_data))
            input_data = jnp.array(input_data)
            target_data = jnp.array(target_data)

            # perform a training step
            loss, model_weights, opt_state = train_step_cnn10(model_weights, model, input_data, target_data, opt_state, optimizer)
            # print or log the loss for monitoring
            print(f"Epoch {epoch + 1}, Loss: {loss}")
        
        # validation loss
        validation_loss = validate(validation_data, model, model_weights)
        val_loss_acc.append(validation_loss)
        print("VALIDTION LOSS:", validation_loss)
        # save best model
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_weights(model_weights, checkpointer, "best_weights")
    # saving last model, hopefully overtrained TODO: actually overtrained??
    save_weights(model_weights, checkpointer, "overtrained_model")
    plt.plot(val_loss_acc)


def validate(validation_data, model, weights):
    total_loss = 0.0
    num_batches = 0
    for batch in validation_data:
        input_data, target_data = batch
        input_data = jnp.array(input_data)
        target_data = jnp.array(target_data)
        prediction = model.apply(weights, input_data)
        loss = loss_fn_cnn10(prediction, target_data)
        total_loss += loss
        num_batches += 1
    return total_loss / num_batches


def save_weights(model_weights, checkpointer, name, folder="/Users/lld/Documents/SoSe 2024/deep_neural_network_analysis/neural_network_analysis/model_checkpoints/"):
    destination = os.path.join(folder, name)
    if os.path.exists(destination):
        # Delete existing folder
        shutil.rmtree(destination)
        print(f"Deleted existing folder: {destination}")
    checkpointer.save(destination, model_weights)
    print(f"Saved weights to: {destination}")


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
    train_loop(train_dataset, validation_dataset, model, weights, optimizer, opt_state)


