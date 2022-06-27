import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# packages for distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn.functional as F


def conv3x3(inplanes, outplanes, stride=1, padding=1):
    """
    Returns a 3x3 convolution layer with padding.
    groups: defines the connection between inputs and outputs. When groups=1, all inputs are convolved to all outputs.
    """
    return nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=3, stride=stride, padding=padding,
                     bias=False)


def conv1x1(inplanes, outplanes, stride=1):
    """
    Returns a 1x1 convolution layer to be used in bottleneck. It is WITHOUT any padding.
    """
    return nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, padding=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, padding=1)
        # print(self.conv1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.skip = nn.Sequential()

        if stride!=1 or inplanes!=self.expansion*planes:
            self.skip = nn.Sequential(
                conv1x1(inplanes, self.expansion*planes, stride),
                nn.BatchNorm2d(self.expansion*planes) )

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # print(out.shape)
        identity = self.skip(x)
        # print(identity.shape)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, self.expansion*planes, stride=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.skip = nn.Sequential()
        if stride!=1 or inplanes != self.expansion*planes:
            self.skip = nn.Sequential(
                conv1x1(inplanes, self.expansion*planes, stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        identity = self.skip(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64

        # the image input layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # calling the in-built function multiple times to copy BasicBlock
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = block.expansion*planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # .size() returns the multiplication of each dimension, .shape returns separate
        out = self.fc(out)

        return out


def ResNet18(classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=classes)


def ResNet50(classes):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=classes)


def ResNet101(classes):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=classes)




# ------ Setting up the distributed environment -------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)



def cleanup():
    dist.destroy_process_group()


def train_model(rank, args):
    print(f"Running Distributed ResNet on rank {rank}.")
    setup(rank, args.world_size)
    torch.manual_seed(0)
    torch.cuda.set_device(rank)

    # instantiate the model and transfer it to the GPU
    model = ResNet101(classes=10).to(rank)
    # wraps the network around distributed package
    model = DDP(model, device_ids=[rank])

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Preparing the training data
    transforms_train = transforms.Compose([transforms.RandomCrop(32, padding=2),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)

    # torch.distributed's own data loader method which loads the data such that they are non-overlapping and
    # exclusive to each process
    train_data_sampler = torch.utils.data.distributed.DistributedSampler(dataset=training_set,
                                                                         num_replicas=args.world_size, rank=rank)
    trainLoader = torch.utils.data.DataLoader(dataset=training_set, batch_size=args.batch_size,
                                              shuffle=False, num_workers=4, pin_memory=True,
                                              sampler=train_data_sampler)

    # Preparing the testing data
    transforms_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    testing_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)

    test_data_sampler = torch.utils.data.distributed.DistributedSampler(dataset=testing_set,
                                                                        num_replicas=args.world_size, rank=rank)
    testLoader = torch.utils.data.DataLoader(dataset=testing_set, batch_size = args.batch_size,
                                             shuffle = False, num_workers=4, pin_memory=True,
                                             sampler=test_data_sampler)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Training
    for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0
        accuracy = 0
        total = 0
        for idx, (inputs, labels) in enumerate(trainLoader):
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            total += labels.size(0)
            _, prediction = outputs.max(1)
            accuracy += prediction.eq(labels).sum().item()

        if rank == 0:
            print("Epoch: {}, Loss: {}, Training Accuracy: {}". format(epoch+1, loss.item(), accuracy/total))

    print("Training DONE!!!")
    print('Testing BEGINS!!')

    # Testing
    test_loss, test_acc, total = 0, 0, 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testLoader):
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, prediction = outputs.max(1)
            total += labels.size(0)
            test_acc += prediction.eq(labels).sum().item()

    # this condition ensures that processes do not trample each other and corrupt the files by overwriting
    if rank == 0:
        print("Loss: {}, Testing Accuracy: {}".format(loss.item(), test_acc / total))
        # Saving the model
        testAccuracy = 100*test_acc/total
        state = {'model': model.state_dict(), 'test_accuracy': testAccuracy, 'num_epochs' : args.n_epochs}
        if not os.path.exists('./models'):
            os.mkdir('./models')
        torch.save(state, './models/cifar10ResNet101.pth')

    cleanup()


def run_train_model(train_func, world_size):

    parser = argparse.ArgumentParser("PyTorch - Training ResNet101 on CIFAR10 Dataset")
    parser.add_argument('--world_size', type=int, default=world_size, help='total number of processes')
    parser.add_argument('--lr', default=0.01, type=float, help='Default Learning Rate')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--n_epochs', type=int, default=150, help='Total number of epochs for training')
    args = parser.parse_args()
    print(args)

    # this is responsible for spawning 'nprocs' number of processes of the train_func function with the given
    # arguments as 'args'
    mp.spawn(train_func, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    # since this example shows a single process per GPU, the number of processes is simply replaced with the
    # number of GPUs available for training.
    n_gpus = torch.cuda.device_count()
    run_train_model(train_model, n_gpus)