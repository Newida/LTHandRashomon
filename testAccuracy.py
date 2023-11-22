from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from resnet20 import Resnet_N_W
from Hparams import Hparams


#setting the path to store/load dataset cifar10
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

milestone_steps='80ep,120ep'
dataset_hparams = Hparams.DatasetHparams()

#(down)load dataset cifar10
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=dataset_hparams.batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=dataset_hparams.batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

plan, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
resnet20model = Resnet_N_W(plan, initializer, outputs)

#setting the path to store/load dataset cifar10
models_path = workdir / "models"
if not data_path.exists():
    data_path.mkdir(parents=True)

resnet20model.load_state_dict(torch.load(models_path / "resnet-20-16_10_7_9.pth"))

correct = 0
total = 0
resnet20model.eval()
for data in testloader:
    images, labels = data
    outputs = resnet20model(images)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print("Accuracy:", 100*correct//total, "%")