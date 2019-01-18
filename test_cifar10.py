import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from resnet import Resnet
import torch.nn.functional as F

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
    # get cifar 10 data
    trainloader, testloader = get_dataset()
    resnet = Resnet()
    resnet.train()
    optimizer = optim.SGD(resnet.parameters(), lr=0.003, momentum=0.0)
    for i, data in enumerate(trainloader, 0):
        for k in range(100):
            x, y = data
            # zero the grad
            optimizer.zero_grad()
            preds = resnet(x)
            loss = F.cross_entropy(preds, y)
            loss.backward()
            optimizer.step()
            if k % 10 == 0:
                print(loss)
        print(f'final loss for one batch: {loss}')
        break

