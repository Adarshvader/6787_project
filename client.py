from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

import femnist
import flwr as fl

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: femnist.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        femnist.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(), len(self.trainloader)

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[int, float, float]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = femnist.test(self.model, self.testloader, device=DEVICE)
        return len(self.testloader), float(loss), float(accuracy)


def main() -> None:
    """Load data, start CifarClient."""

    # Load model and data
    model = femnist.Net()
    model.to(DEVICE)
    trainloader, testloader = femnist.load_data()

    # Start client
    client = CifarClient(model, trainloader, testloader)
    fl.client.start_numpy_client("0.0.0.0:8080", client)


if __name__ == "__main__":
    main()
