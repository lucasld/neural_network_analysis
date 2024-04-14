import tensorflow as tf
import tensorflow_datasets as tfds

def load_cifar10(batch_size=32, shuffle_buffer_size=10000, seed=None):
    # Set seed
    tf.random.set_seed(seed)

    # Load CIFAR-10 dataset
    train_ds, val_ds, test_ds = tfds.load('cifar10', split=['train', 'test[:50%]', 'test[50%:]'])
    
    # Define preprocessing function
    def preprocess_dataset(dataset):
        def preprocess_data(sample):
            image = tf.cast(sample['image'], tf.float32) / 255.0  # Normalize pixel values to [0, 1]
            label = sample['label']
            return image, label
        
        dataset = dataset.map(preprocess_data)
        # Shuffle the data
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
        # Batch and prefetch the data
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
    
    # Apply preprocessing
    train_dataset = preprocess_dataset(train_ds)
    validation_dataset = preprocess_dataset(val_ds)
    test_dataset = preprocess_dataset(test_ds)

    return train_dataset, validation_dataset, test_dataset


print(load_cifar10())