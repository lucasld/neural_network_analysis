# Deep Neural Network Analysis
This repository serves as a resource for implementing and understanding deep neural network analysis techniques. Here, you'll find pre-defined models with weights for untrained, trained, and overfitted scenarios. The repository includes both an image classifier and a regression network. The repository also contains some [slides](presentation.pdf) going into detail about Xavier and Kaiming weight intilaization.

## Project Structure
### `model_checkpoints/`
Here you can find all the models weights. Each of the subfolders contains untrained weights, overtrained weights and the best performing weights from the training run. You can also find a simple plot showing the training loss and the validation loss.

**Model Configurations:**
1. **CIFAR-10 Dataset**:
    - **Xavier Uniform Initialization**:
        - [Tanh Activation](model_checkpoints/cifar10/xavier_uniform/tanh/)
        - [ReLU Activation](model_checkpoints/cifar10/xavier_uniform/relu/)
        - [Sigmoid Activation](model_checkpoints/cifar10/xavier_uniform/sigmoid/)
    - **Kaiming Normal Initialization**:
        - [Tanh Activation](model_checkpoints/cifar10/kaiming_normal/tanh/)
        - [ReLU Activation](model_checkpoints/cifar10/kaiming_normal/relu/)
        - [Sigmoid Activation](model_checkpoints/cifar10/kaiming_normal/sigmoid/)
    - **Kaiming Uniform Initialization**:
        - [Tanh Activation](model_checkpoints/cifar10/kaiming_uniform/tanh/)
        - [ReLU Activation](model_checkpoints/cifar10/kaiming_uniform/relu/)
        - [Sigmoid Activation](model_checkpoints/cifar10/kaiming_uniform/sigmoid/)
2. **Wine Quality Dataset**:
    - **Xavier Uniform Initialization**:
        - [Tanh Activation](model_checkpoints/wine_quality/xavier_uniform/tanh/)
        - [ReLU Activation](model_checkpoints/wine_quality/xavier_uniform/relu/)
        - [Sigmoid Activation](model_checkpoints/wine_quality/xavier_uniform/sigmoid/)
    - **Kaiming Normal Initialization**:
        - [Tanh Activation](model_checkpoints/wine_quality/kaiming_normal/tanh/)
        - [ReLU Activation](model_checkpoints/wine_quality/kaiming_normal/relu/)
        - [Sigmoid Activation](model_checkpoints/wine_quality/kaiming_normal/sigmoid/)
    - **Kaiming Uniform Initialization**:
        - [Tanh Activation](model_checkpoints/wine_quality/kaiming_uniform/tanh/)
        - [ReLU Activation](model_checkpoints/wine_quality/kaiming_uniform/relu/)
        - [Sigmoid Activation](model_checkpoints/wine_quality/kaiming_uniform/sigmoid/)

### `src/`
This folder contains all the code related to loading the datasets, creating the models and training them.


### `main.py`
This file can be run to train a model using a certain configuration.

```bash
python3 main.py [task] [init method] [activation func]
```
* **task:** cifar10 / wine_quality
* **init method:** xavier_uniform / kaiming_normal / kaiming_uniform
* **activation func:** tanh / relu / sigmoid


### `main.sh`
Executes *main.py* for all configurations.


## How to Use
### Installation

**(version of Python == 3.11)**

1. Clone the repository:

   ```bash
   git clone https://github.com/lucasld/neural_network_analysis.git
   ```

2. Navigate to the project directory:
   ```bash
   cd neural_network_analysis
   ```

3. Install the required packages (possibly activate/create conda env):
    * Run the following command to create a virtual environment:
        ```bash
        python3 -m venv myenv
        ```
    * Activate the virtual environment:
        - **Windows**:
            ```
            myenv\Scripts\activate
            ```
        - **Unix/Linux/macOS**:
            ```
            source myenv/bin/activate
            ```
    * Install project dependencies
        ```bash
        pip install -r requirements.txt
        ```

4. If creating the env from the environment file did not work for you,
you can install all dependencies via:

    Run one of the 3 commands for cpu, cuda or tpu support respectively:
    ```
    pip install -U "jax[cpu]"
    pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    ```
    Run the rest:
    ```
    pip install optax==0.1.7
    ```
    ```
    pip install scipy==1.11.4
    ```
    ```
    pip install -U "flax[all]"
    ```
    ```
    pip install tensorflow_datasets
    ```
    ```
    pip install tensorflow
    ```
    ```
    pip install matplotlib
    ```
5. To run all experiments run:
    ```
    sh main.sh
    ```

    To run a single experiment run:
   ```
    python main.py cifar10|wine_quality xavier_uniform|glorot_uniform|glorot_normal tanh|relu|sigmoid
   ```
   
   
   

### Loading weights
Load the weights using the ```load_weights()``` function located in [src/utils.py](src/utils.py).


