import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from resnet import Resnet
import sys

def get_dataset():
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)
    return trainloader, testloader 

def get_test_accuracy(net, testloader):
  accs = []
  net.eval()
  with torch.no_grad():
    for i, data in enumerate(testloader, 0):
      x, y = data
      if use_cuda:
        x, y = x.cuda(), y.cuda()
      preds = net(x)
      _, predicted = torch.max(preds, 1)
      accuracy = accuracy_score(predicted.cpu() if use_cuda else predicted, y.cpu() if use_cuda else y)
      accs.append(accuracy)
      break
  net.train()
  return np.mean(accs)

def update_learning_rate(current_lr, optimizer):
  new_lr = current_lr/10
  for g in optimizer.param_groups:
    g['lr'] = new_lr
  return new_lr



if __name__ == '__main__':
    use_cuda = len(sys.argv) > 1 and sys.argv[1] == 'cuda'
    num_epochs = 80
    # get cifar 10 data
    trainloader, testloader = get_dataset()
    benchmark, debug = False, True
    resnet = Resnet(n=2,dbg=debug)
    resnet.train()
    if use_cuda:
        resnet = resnet.cuda()
        for block in resnet.residual_blocks:
            block.cuda()
    current_lr = 1e-4
#     optimizer = optim.SGD(resnet.parameters(), lr=current_lr, weight_decay=0.0001, momentum=0.9)
    optimizer = optim.Adam(resnet.parameters(), lr=1e-4, weight_decay=0.0001)
    train_accs, test_accs = [], []
    gradient_norms = []
    def train_model():
      current_lr=1e-4
      stopping_threshold, current_count = 3, 0
      n_iters = 0
      for e in range(num_epochs):
        # modify learning rate at 
          for i, data in enumerate(trainloader, 0):
              x, y = data
              if use_cuda:
                x, y = x.cuda(), y.cuda()
              # zero the grad
              optimizer.zero_grad()
              preds = resnet(x)
              loss = F.cross_entropy(preds, y)
              loss.backward()
              optimizer.step()
              if i % 10 == 0:
                  _, predicted = torch.max(preds, 1)
                  accuracy = accuracy_score(predicted.cpu() if use_cuda else predicted, y.cpu() if use_cuda else y)
                  train_accs.append(accuracy)
                  print('n_iters: {} loss: {}, accuracy: {}'.format(n_iters, loss, accuracy))
              if i % 50 == 0:
                # get test accuracy
                test_acc = get_test_accuracy(resnet, testloader)
                test_accs.append(test_acc)
                # monitor gradient norms
                total_norm = sum([p.grad.data.norm(2).item() ** 2 for p in resnet.parameters()])
                total_norm**=(1./2)
                gradient_norms.append(total_norm)
                print('n_iters: {} test accuracy: {} gradient_norm: {}'.format(n_iters, test_acc, total_norm))
              n_iters+=1
              if n_iters == 10000 or n_iters == 16000 or n_iters == 24000:
                current_lr = update_learning_rate(current_lr, optimizer)
                print('decayed learning rate to {}'.format(current_lr))
      print('iterated {} times'.format(n_iters))
      return resnet, n_iters, train_accs, test_accs
    
    trained_resnet, n_iters, train_accs, test_accs, gradient_norms = train_model()
    print(len(train_accs), len(test_accs), len(gradient_norms))
    plt.plot(range(len(train_accs)), train_accs)
    plt.plot(range(len(test_accs)), test_accs)
    plt.plot(range(len(gradient_norms)), gradient_norms)
    plt.show()

    
                

