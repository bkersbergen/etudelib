import torch
import torch.nn as nn
import torch.optim as optim


# Define the neural network architecture
# Define a simple convolutional neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Set up the device and load the dataset
devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')


for device in devices:
    # Initialize the model and optimizer
    model = Net().to(device)
    model.eval()

    # Convert the model to a JIT format and save it to disk
    example_input = torch.rand(1, 1, 28, 28).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(f'models/mnist_{device}.pt')
