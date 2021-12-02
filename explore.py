import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_datasets as tfds
import collections

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
    only_digits=True, cache_dir=None)

example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])

print(len(example_dataset))

NUM_EPOCHS = 5
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

def preprocess(dataset, batch_size):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 28,28]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.shuffle(SHUFFLE_BUFFER, seed=1).batch(
      batch_size).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

processed = preprocess(example_dataset, 93)
sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(processed)))

print(sample_batch['x'][0])
