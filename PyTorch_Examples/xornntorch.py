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
class Net_run():
    def __init__(self,model_arch,inputs=None,targets=None):
        self.net = Net()
        if inputs==None:
            self.inputs = list(map(lambda s: Variable(torch.Tensor([s])), [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ]))
        if targets == None:
            self.targets = list(map(lambda s: Variable(torch.Tensor([s])), [
                [0],
                [1],
                [1],
                [0]
            ])) 
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001)
    def train(self,criterion=None,optimizer=None):
        for i in range(0, 10):
            for input, target in zip(self.inputs, self.targets):
                self.optimizer.zero_grad()   
                output = self.net(input)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()  
            print("Loss: {}".format(loss.data[0]))
    def test(self):
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
    net = Net()
    n = Net_run(net)
    n.train()
    n.test()
