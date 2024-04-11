import tensorflow as tf
import tensorflow_datasets as tfds

# TODO: SEED
# TODO: SPLIT

def load_cifar10(batch_size=32, shuffle_buffer_size=10000, splits=[0.7, 0.2, 0.1]):
    # Load CIFAR-10 dataset
    cifar10_dataset, info = tfds.load('cifar10', split='train', with_info=True)
    
    # Define preprocessing function
    def preprocess_data(sample):
        image = tf.cast(sample['image'], tf.float32) / 255.0  # Normalize pixel values to [0, 1]
        label = sample['label']
        return image, label
    
    # Apply preprocessing function and shuffle the data
    cifar10_dataset = cifar10_dataset.map(preprocess_data)
    cifar10_dataset = cifar10_dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    # Batch and prefetch the data
    cifar10_dataset = cifar10_dataset.batch(batch_size)
    cifar10_dataset = cifar10_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return cifar10_dataset, info