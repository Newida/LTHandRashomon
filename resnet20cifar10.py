from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from resnet20 import Resnet_N_W
from Hparams import Hparams
from utils_DataLoader import DataLoaderHelper


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

#hyper parameters taken from unmaksing the lth 
#training hyperparameters
#optimizer_name='sgd'
#momentum=0.9
#milestone_steps='80ep,120ep'
#lr=0.1
#gamma=0.1
#weight_decay=1e-4
#training_steps='160ep'
#data_order_seed = 0
#pruning hyperparemeters
#pruning_fraction = 0.2
#dataset hyperparameters
#batch_size = 128
#model hyperparameters
#model_initializer='kaiming_normal'
#batchnorm_init='uniform' TODO: What does that mean? How is this implemented by frankle? 
#rewind_points = [0, 250, 2000]
#loss = CrossEntropyLoss

dataset_hparams = Hparams.DatasetHparams()

#(down)load dataset cifar10
dataloaderhelper = DataLoaderHelper(0, 0, dataset_hparams)

trainset = dataloaderhelper.get_trainset(data_path, transform)
testset = dataloaderhelper.get_testset(data_path, transform)

trainset, valset = dataloaderhelper.split_train_val(trainset)

testloader = dataloaderhelper.get_test_loader(testset)
valloader = dataloaderhelper.get_validation_loader(valset)
trainloader = dataloaderhelper.get_train_loader(trainset)

#strategy:
#1. define seed
#2. split training data into training and validation
#3. use training data as before
#4. add early stopping using the validation dataset
#dont shuffle validation set
#shuffle training data
#create a class or something to keep everything the same, when using the seed
#that means validation set and shuffling during training should stay the same given a seed

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#create model
plan, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
resnet20model = Resnet_N_W(plan, initializer, outputs)

#initialize hyperparemeters
model_hparams = Hparams.ModelHparams()
training_hparams = Hparams.TrainingHparams(num_epoch=20, milestone_steps=[10, 15])
pruning_hparams = Hparams.PruningHparams() #not used yet

#do training 
#TODO: be wary for the randomness in the training
# as it is important to identify different winning tickets later
# therefore add sampler that allows to influence advanced shuffeling
def train(model, model_hparams, training_hparams):
    model.to(device)
    model.train()
    #not implemented yet
    optimizer = Hparams.get_optimizer(model, training_hparams)
    lr_scheduler = Hparams.get_lr_scheduler(optimizer, training_hparams)
    loss_criterion = Hparams.get_loss_criterion(training_hparams)

    #data_order_seed = training_hparams.data_order_seed not used yet

    #implement early stopping instead
    print("Started training ...")
    for epoch in range(training_hparams.num_epoch):
        trainloader.shuffle(trainloader.get_seed() + epoch)
        #shuffle data for each epoch,
        #usually done by setting shuffle = True in the dataloader
        #but not in our case since we have a custom one
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = resnet20model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        lr_scheduler.step()
        
models_path = workdir / "models"
if not models_path.exists():
    models_path.mkdir(parents=True)

import time
start = time.time()
train(resnet20model, model_hparams, training_hparams)
end = time.time()
print("Time of training:", end - start)
torch.save(resnet20model.state_dict(), models_path / "resnet-20-16_20_10_15.pth")