import jax
from functools import partial
import optax
import jax.numpy as jnp
import os

from src.utils import save_weights, plot_validation_losses, progress_print


@jax.jit
def loss_fn_cnn10(predictions, target_data):
    # Compute softmax cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(predictions,target_data)
    return jnp.mean(loss)


@jax.jit
def loss_fn_wine(predictions, target_data):
    # Compute softmax cross-entropy loss
    loss = optax.squared_error(predictions,target_data)
    return jnp.mean(loss)


@partial(jax.jit, static_argnums=[1, 4])
@partial(jax.value_and_grad, argnums=0)
def forward(weights, model, input_data, target_data, loss_fn):
    prediction = model.apply(weights, input_data)
    return loss_fn(prediction, target_data)


@partial(jax.jit, static_argnums=[1, 5, 6])
def train_step(weights, model, input_data, target_data, opt_state, optimizer, loss_fn):
    loss_v, grads = forward(weights, model, input_data,target_data, loss_fn)

    updates, opt_state = optimizer.update(grads, opt_state)
    weights = optax.apply_updates(weights, updates)

    return loss_v, weights,opt_state


def train_loop(training_data, validation_data, model, model_weights, optimizer,
               opt_state, loss_fn, checkpoint_path, epochs, plot_every=1):
    """Training loop for model. Also saves weights before training, after
    training and best performing weights.
    
    :param training_data: data the model should be trained on
    :type training_data:
    :param validation_data: data used for validation
    :type validation_data:
    :param model: model object
    :type model:
    :param model_weights: weights and biases for the model
    :type model_weights:
    :param optimizer: optimizer used to calculate weight updates
    :type optimizer:
    :param opt_state: optimizer parameters TODO:??
    :type opt_state:
    :param loss_fn: loss function
    :type loss_fn:
    :param checkpoint_path: path pointing to folder where weight-checkpoints
        should be saved, if False/None nothing will be saved
    :type checkpoint_path: str
    :param epochs: number epochs for which to train
    :type epochs: int
    """
    val_losses = []
    train_losses = []
    # save initial weights
    if checkpoint_path:
        save_weights(model_weights, checkpoint_path, "initial_weights")

    # training
    for epoch in range(epochs):
        # iterate over batches of training data
        losses = []
        for batch in training_data:
            input_data, target_data = batch
            input_data = jnp.array(input_data)
            target_data = jnp.array(target_data)
            # perform a training step
            loss, model_weights, opt_state = train_step(model_weights, model, input_data, target_data, opt_state, optimizer, loss_fn)
            # print loss for monitoring
            losses.append(loss)
            progress_print(epoch, loss)
        # validation loss
        validation_loss = validate(validation_data, model, model_weights, loss_fn)
        progress_print(epoch, loss, validation_loss)
        # save best model
        if not val_losses or validation_loss < min(val_losses):
            if checkpoint_path:
                save_weights(model_weights, checkpoint_path, "best_weights")
        val_losses.append(validation_loss)
        train_losses.append(sum(losses)/len(losses))
        losses = []
        if epoch%plot_every == 0:
            plot_validation_losses(val_losses, train_losses)
    # save validation loss plot
    plot_path = os.path.join(checkpoint_path, 'validation_losses.png')
    plot_validation_losses(val_losses, train_losses, plot_path)

    # saving last model, hopefully overtrained TODO: actually overtrained??
    if checkpoint_path:
        save_weights(model_weights, checkpoint_path, "overtrained_model")


def validate(validation_data, model, weights, loss_fn) -> float:
    """Calculate validation loss for model and weights.
    
    :param validation_data: validation data batches
    :type validation_data:
    :param model: model structure
    :type model:
    :param weights: weights to be applied to model
    :type weights:
    :returns: validation data loss for model with weights
    :rtype: float
    """
    total_loss = 0.0
    num_batches = 0
    for batch in validation_data:
        input_data, target_data = batch
        input_data = jnp.array(input_data)
        target_data = jnp.array(target_data)
        loss_v, _ = forward(weights, model, input_data, target_data, loss_fn)
        total_loss += loss_v
        num_batches += 1
    return total_loss / num_batches