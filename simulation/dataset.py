# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Partitioned version of CIFAR-10 dataset."""

from typing import List, Tuple, cast

import numpy as np
import tensorflow as tf
import collections
import tensorflow_federated as tff

""" XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]


def shuffle(x: np.ndarray, y: np.ndarray) -> XY:
    Shuffle x and y
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


def partition(x: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    Split x and y into a number of partitions
    return list(zip(np.split(x, num_partitions), np.split(y, num_partitions)))


def create_partitions(
    source_dataset: XY,
    num_partitions: int,
) -> XYList:
    Create partitioned version of a source dataset.
    x, y = source_dataset
    x, y = shuffle(x, y)
    xy_partitions = partition(x, y, num_partitions)

    return xy_partitions"""


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


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""

    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    client_train = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[idx])
    client_test = emnist_test.create_tf_dataset_for_client(emnist_train.client_ids[idx])

    processed_train = preprocess(client_train, len(list(client_train)))
    processed_test = preprocess(client_test, len(list(client_test)))

    sample_train = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(processed_train)))

    sample_test = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(processed_test)))

    x_train = sample_train['x']
    y_train = sample_train['y']

    x_test = sample_test['x']
    y_test = sample_test['y']

    return (x_train,y_train),(x_test,y_test)

def load_partitions():
    """Load 1/10th of the training and test data to simulate a partition."""

    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    xy_train_partitions = []
    xy_test_partitions = []
    ids = emnist_train.client_ids

    print(len(ids))

    for id in range(0, 100):
        client_train = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[id])
        client_test = emnist_test.create_tf_dataset_for_client(emnist_train.client_ids[id])

        processed_train = preprocess(client_train, len(list(client_train)))
        processed_test = preprocess(client_test, len(list(client_test)))

        sample_train = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(processed_train)))

        sample_test = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(processed_test)))

        x_train = sample_train['x']
        y_train = sample_train['y']

        x_test = sample_test['x']
        y_test = sample_test['y']

        xy_train_partitions.append((x_train, y_train))
        xy_test_partitions.append((x_test, y_test))

    return list(zip(xy_train_partitions, xy_test_partitions))


"""def load(
    num_partitions: int,
) -> PartitionedDataset:
    Create partitioned version of CIFAR-10
    xy_train, xy_test = tf.keras.datasets.fashion_mnist.load_data()

    xy_train_partitions = create_partitions(xy_train, num_partitions)
    xy_test_partitions = create_partitions(xy_test, num_partitions)

    return list(zip(xy_train_partitions, xy_test_partitions))"""
