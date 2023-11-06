import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Subset, SubsetRandomSampler
from torchvision.models import resnet34, ResNet34_Weights

classes_to_learn = [0, 1, 2]
num_epochs = 10
learning_rate = 0.001
momentum = 0.9
batch_size = 64
teacher = resnet34(weights=ResNet34_Weights.DEFAULT)
teacher_name = "resnet34"

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

train_indices = [i for i in range(len(train_set)) if train_set[i][1] in classes_to_learn]
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=2,
                                          sampler=SubsetRandomSampler(train_indices))

# Create a teacher-34 model
num_ftrs = teacher.fc.in_features
teacher.fc = nn.Linear(num_ftrs, len(classes_to_learn))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(teacher.parameters(), lr=learning_rate, momentum=momentum)

# Train the model
if __name__ == '__main__':
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / 100:.4f}')

    print('Finished Training')
    classes = '_'.join([f"{i}" for i in classes_to_learn])
    torch.save(teacher.state_dict(), f"teacher_{teacher_name}_classes_{classes}.weights")
