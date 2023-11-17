from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms


#setting the path to store/load dataset
workdir = Path.cwd()
data_path = workdir / "datasets" / "cifar10"
if not data_path.exists():
    data_path.mkdir(parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data augmentation
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean, std)
     ])

#hyper parameters taken from unmaksing the lth 
#training hyperparameters
optimizer_name='sgd'
momentum=0.9
milestone_steps='80ep,120ep'
lr=0.1
gamma=0.1
weight_decay=1e-4
training_steps='160ep'
#pruning hyperparemeters
pruning_fraction = 0.2
#dataset hyperparameters
batch_size = 128
#model hyperparameters
model_initializer='kaiming_normal'
#batchnorm_init='uniform' TODO: What does that mean? How is this implemented by frankle? 
#rewind_points = [0, 250, 2000]
#loss = CrossEntropyLoss

#(down)load dataset cifar10
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')