import os
import pickle
import matplotlib.pyplot as plt



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
    :returns: models weights
    :rtype:
    """
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def plot_validation_losses(validation_losses: list, save_path: str=None):
    """
    Plots the validation losses using matplotlib and saves the plot.

    
    :param validation_losses: list of validation losses
    :type validation_losses: list
    :param save_path: path where the plot will be saved, if None plot will not
        be saved, defaults to None
    :type save_path: str, optional
    """
    # Create x-axis values (epochs)
    epochs = range(1, len(validation_losses) + 1)

    # Plot validation losses
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    if save_path:
        # Save the plot
        plt.savefig(save_path)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def progress_print(epoch: int, train_loss: float, val_loss: float=None):
    print('\r', end='')
    print(f"Epoch {epoch} -- train loss: {train_loss} {'-- val loss: {}'.format(val_loss) if val_loss else ''}", end='', flush=True)
    if val_loss:
        print()