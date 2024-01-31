# from https://github.com/kuangliu/pytorch-cifar/tree/master
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from resnet_model import ResNet50
import os


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving.. ', acc)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


"""
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')"""

if __name__ == '__main__':
    lr = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    num_epochs = 50
    batch_size = 64
    classes_to_learn = [0, 1, 2]
    model_name = "resnet50"

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_indices = [i for i in range(len(train_set)) if train_set[i][1] in classes_to_learn]

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=2,
                                              sampler=SubsetRandomSampler(train_indices))

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_indices = [i for i in range(len(test_set)) if test_set[i][1] in classes_to_learn]
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=2,
                                             sampler=SubsetRandomSampler(test_indices))

    # Model
    print('==> Building model..')
    net = ResNet50(len(classes_to_learn))
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(num_epochs):
        train(epoch)
        test(epoch)
        scheduler.step()

    print('Finished Training')
    classes = '_'.join([f"{i}" for i in classes_to_learn])
    torch.save(net.state_dict(), f"teacher_new_{model_name}_classes_{classes}.weights")
