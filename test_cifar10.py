import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from resnet import Resnet
import example_resnet
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import argparse

def get_dataset():
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)
    return trainloader, testloader 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet implementation')
    parser.add_argument('--debug', action='store_true', default=False, help='run in debug mode, basically prints out a lot of shapes for development help')
    parser.add_argument('--benchmark', action='store_true', default=False, help='use example ResNet to benchmark my implememtation')
    args = parser.parse_args()
    num_epochs = 5
    # get cifar 10 data
    trainloader, testloader = get_dataset()
    if not args.benchmark:
        print('Using my resnet')
        resnet = Resnet(dbg=args.debug)
    else:
        print('Using benchmark resnet')
        resnet = example_resnet.ResNet18()
    resnet.train()
    optimizer = optim.SGD(resnet.parameters(), lr=0.03, momentum=0.0)
    for e in range(num_epochs):
        for i, data in enumerate(trainloader, 0):
            if i == 5:
                break
            print(i)
            x, y = data
            # zero the grad
            optimizer.zero_grad()
            preds = resnet(x)
            loss = F.cross_entropy(preds, y)
            loss.backward()
            optimizer.step()
            if i % 1 == 0:
                _, predicted = torch.max(preds, 1)
                accuracy = accuracy_score(predicted, y)
                print(f'loss: {loss}, accuracy: {accuracy}')

