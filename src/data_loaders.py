import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def load_cifar10(batch_size=32, shuffle_buffer_size=10000, seed=42):
    """Loads and preprocesses cifar10 dataset using tensorflow_datasets.
    
    :param batch_size: batch size of the resulting datasets, defaults to 32
    :type batch_size: int, optional
    :param shuffle_buffer_size: size of the buffer used for shuffling the dataset, defaults to 10000
    :type shuffle_buffer_size: int, optional
    :param seed: random seed for shuffling, defaults to None
    :type seed: int, optional
    :return: Tuple of (train_dataset, test_dataset)
    :rtype: tuple
    """
    # set seed
    tf.random.set_seed(seed)
    # load CIFAR-10 dataset
    train_ds, val_ds, test_ds = tfds.load('cifar10', split=['train', 'test[:50%]', 'test[50%:]'])
    
    # define preprocessing function
    def preprocess_dataset(dataset):
        def preprocess_data(sample):
            image = tf.cast(sample['image'], tf.float32) / 255.0  # Normalize pixel values to [0, 1]
            label = sample['label']
            return image, label
        
        dataset = dataset.map(preprocess_data)
        # shuffle the data
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
        # Batch and prefetch the data
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
    
    # apply preprocessing
    train_dataset = preprocess_dataset(train_ds)
    validation_dataset = preprocess_dataset(val_ds)
    test_dataset = preprocess_dataset(test_ds)

    return train_dataset, validation_dataset, test_dataset


def load_wine_quality(batch_size=32, shuffle_buffer_size=10000, seed=42):
    """Loads and preprocesses wine_quality dataset using tensorflow_datasets.
    
    :param batch_size: batch size of the resulting datasets, defaults to 32
    :type batch_size: int, optional
    :param shuffle_buffer_size: size of the buffer used for shuffling the dataset, defaults to 10000
    :type shuffle_buffer_size: int, optional
    :param seed: random seed for shuffling, defaults to None
    :type seed: int, optional
    :return: tuple of (train_dataset, val_dataset, test_dataset)
    :rtype: tuple
    """
    # set seed
    tf.random.set_seed(seed)
    # load White Wine Quality dataset
    dataset = tfds.load('wine_quality/white', split='train')
    
    # split the dataset into train, validation, and test sets
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size).take(test_size)

    # extract feature values from the dataset and calculate mean and std_dev
    feature_values = []
    for example in train_dataset.as_numpy_iterator():
        feature_values.append(np.array(list(example['features'].values())))
        # scale quality value from 1 to 10 to 0 to 1
    feature_values = np.array(feature_values)

    # calculate mean and std
    mean = np.mean(feature_values, axis=0)
    std_dev = np.std(feature_values, axis=0)

    # apply shuffling, batching and prefetching
    def preprocess_dataset(dataset):
        # change datset format
        def stack(x):
            features = list(x['features'].values())
            cast_features = [tf.cast(v, tf.float32) for v in features]
            return tf.stack(cast_features)
        dataset = dataset.map(lambda x : (stack(x), x["quality"]))
        # normalize
        dataset = dataset.map(lambda x,y : ((x-mean)/std_dev, tf.expand_dims(y/10, 0)))
        # shuffle the data
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
        # batch and prefetch the data
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
    
    train_dataset = preprocess_dataset(train_dataset)
    val_dataset = preprocess_dataset(val_dataset)
    test_dataset = preprocess_dataset(test_dataset)

    return train_dataset, val_dataset, test_dataset
