import torch
import torch.nn as nn
import torchvision
from typing import Tuple

class  Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #first conv layer
        self.cnn_layer1 = torch.nn.Sequential(

            torch.nn.Conv2d(1, 32, 3, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=1))

        #second conv layer
        self.cnn_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 3, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=1))


        #linear layer
        self.linear_layers = torch.nn.Sequential(

            #dense layer
            torch.nn.Linear(15488,128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128,10)
        )


    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

mnist_path = 'path/to/mnist/data'

def load_data() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Load mnist (training and test set)."""
    transform = torchvision.transforms.ToTensor()

    trainset = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testset = torchvision.datasets.MNIST(root=mnist_path, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    return trainloader, testloader


def train(net: Net, trainloader: torch.utils.data.DataLoader, epochs: int,
    device: torch.device,) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0





def test(net: Net, testloader: torch.utils.data.DataLoader,
device: torch.device,) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader = load_data()
    print("Start training")
    net=Net().to(DEVICE)
    train(net=net, trainloader=trainloader, epochs=2, device=DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()
