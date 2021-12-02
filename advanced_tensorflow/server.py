from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf
from client import preprocess
import numpy as np
import tensorflow_federated as tff
import collections



def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)])

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_eval=1,
        min_fit_clients=3,
        min_eval_clients=2,
        min_available_clients=10,
        eval_fn=get_eval_fn(model, 10),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": 10}, strategy=strategy)


def get_eval_fn(model, num_clients):
    """Return an evaluation function for server-side evaluation."""

    x_vals = np.empty((0,28,28))
    y_vals = np.empty((0,1))

    for i in range(num_clients):
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
        client_test = emnist_test.create_tf_dataset_for_client(emnist_train.client_ids[i])


        processed_client_test = preprocess(client_test, len(list(client_test)))

        sample_test = tf.nest.map_structure(lambda x: x.numpy(),
                                         next(iter(processed_client_test)))


        x_vals = np.append(x_vals,sample_test['x'], axis=0)
        y_vals = np.append(y_vals,sample_test['y'], axis=0)


    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_vals, y_vals)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
