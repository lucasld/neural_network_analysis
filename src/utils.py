import matplotlib.pyplot as plt
import os
import pickle


def create_folder(path: str):
    """Creates folder from path.
    
    :param path: path to folder that should be created
    :type path: str
    """

    os.makedirs(path,exist_ok=True)


def save_weights(weights, path: str, name: str):
    """Save weights.

    :param weights: weights to be saved
    :type weights: 
    :param path: path to the folder the weights should be saved in
    :type folder_path: str
    :param name: name of the file to be created (without .pkl ending)
    :type name: str
    """
    filepath = os.path.join(path, name) + '.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(weights, f)


def load_weights(path: str):
    """Load weights from path.
    
    :param path: path that werights are saved at
    :type path: str
    :return: models weights
    :rtype:
    """
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def plot_validation_losses(validation_losses: list, train_losses: list,
                           save_path: str=None):
    """
    Plots the validation losses using matplotlib and saves the plot.
    
    :param validation_losses: list of validation losses
    :type validation_losses: list
    :param train_losses: list of training losses
    :type train_losses: list
    :param save_path: path where the plot will be saved, if None plot will not
        be saved, defaults to None
    :type save_path: str, optional
    """
    # Create x-axis values (epochs)
    epochs = range(1, len(validation_losses) + 1)
    # Plot validation losses
    plt.close()
    if validation_losses:
        plt.plot(epochs, validation_losses, label='Validation Loss')
    if train_losses:
        plt.plot(epochs, train_losses, label='Training Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        # Save the plot
        plt.savefig(save_path)
    plt.show(block=False)
    plt.pause(0.1)


def progress_print(epoch: int, train_loss: float, val_loss: float=None):
    """Progressbar to output training progress parameters to console.
    
    :param epoch: current epoch
    :type epoch: int
    :param train_loss: current training loss
    :type train_loss: float
    :param val_loss: current validation loss, defaults to None
    :type val_loss: float, optional
    """
    print('\r', end='')
    print(f"Epoch {epoch} -- train loss: {train_loss} {'-- val loss: {}'.format(val_loss) if val_loss else ''}", end='', flush=True)
    if val_loss:
        print()