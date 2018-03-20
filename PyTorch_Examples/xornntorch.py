import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 5, True)
        self.fc2 = nn.Linear(5, 1, True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
class net_run():
    def __init__(self):
        self.net = Net()
        self.inputs = list(map(lambda s: Variable(torch.Tensor([s])), [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]))
        self.targets = list(map(lambda s: Variable(torch.Tensor([s])), [
            [0],
            [1],
            [1],
            [0]
        ]))
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001)
    def run(self):
        for i in range(0, 10000):
            for input, target in zip(self.inputs, self.targets):
                self.optimizer.zero_grad()   
                output = self.net(input)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()  
                if i % 500 == 0:
                    print(loss)
        print("Training Complete")
        for input, target in zip(self.inputs, self.targets):
            output = self.net(input)
            print("Input:[{},{}] Target:[{}] Predicted:[{}] Error:[{}]".format(
                int(input.data.numpy()[0][0]),
                int(input.data.numpy()[0][1]),
                int(target.data.numpy()[0]),
                (float(output.data.numpy()[0]), 4),
                (float(abs(target.data.numpy()[0] - output.data.numpy()[0])), 4)
            ))

if __name__ == "__main__":
    n = net_run()
    n.run()
