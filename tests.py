import torch
import numpy as np
from resnet20 import Resnet_N_W
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from Hparams import Hparams
from utils import DataLoaderHelper
from utils_Earlystopper import EarlyStopper
import routines

#test is_valid_model_name:
#false cases
print("Testing is_valid_model_name:")
print("-False cases:")
print(Resnet_N_W.is_valid_model_name("resnet"))
print(Resnet_N_W.is_valid_model_name("resnet-"))
print(Resnet_N_W.is_valid_model_name("resnet-18"))
print(Resnet_N_W.is_valid_model_name("resnet-2-20"))

#true cases
print("-True cases:")
print(Resnet_N_W.is_valid_model_name("resnet-20"))
print(Resnet_N_W.is_valid_model_name("resnet-20-16"))
print(Resnet_N_W.is_valid_model_name("resnet-20-24"))

print("-"*20)
#test is_valid_initalizer:
#false cases
print("Testing is_valid_initalizer")
print("-False cases:")
print(Resnet_N_W.is_valid_initalizer("resnet"))

#true cases
print("-True cases:")
print(Resnet_N_W.is_valid_initalizer("kaiming_normal"))
print(Resnet_N_W.is_valid_initalizer("kaiming_uniform"))

print("-"*20)
#test get_model_from_name:
print("Testing get_model_from_name")
#regular cases
print("-Testing regular cases:")
print(Resnet_N_W.get_model_from_name("resnet-20"))
print(Resnet_N_W.get_model_from_name("resnet-20", "kaiming_normal"))
print(Resnet_N_W.get_model_from_name("resnet-20-16", "kaiming_uniform"))
print(Resnet_N_W.get_model_from_name("resnet-20-16", "kaiming_normal", outputs=100))
print(Resnet_N_W.get_model_from_name("resnet-32"))
print(Resnet_N_W.get_model_from_name("resnet-44"))
print(Resnet_N_W.get_model_from_name("resnet-56"))

#error cases
print("Testing error inputs:")
try:
    print(Resnet_N_W.get_model_from_name("resnet-18-10"))
except ValueError:
    print("ValueError raised")
try:
    print(Resnet_N_W.get_model_from_name("resnet-20", "somethingelse"))
except ValueError:
    print("ValueError raised")

print("-"*20)

#test Resnet:
print("Testing Resnet:")
print("-Testing structure")
plan, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
model_hparams = Hparams.ModelHparams(plan, initializer, outputs, 0)
resnet20model = Resnet_N_W(model_hparams)
print(resnet20model.blocks)
print("-Testing forwardstep")
input = torch.unsqueeze(torch.stack([torch.eye(32),torch.eye(32),torch.eye(32)]), 0)
resnet20model(input)
print("-"*20)

#Test reintialization
plan, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
model_hparams = Hparams.ModelHparams(plan, initializer, outputs, 0)
resnet20model = Resnet_N_W(model_hparams)
resnet20model_copy = Resnet_N_W(model_hparams)
resnet20model_copy.load_state_dict(resnet20model.state_dict())
resnet20model_untouched = Resnet_N_W(model_hparams)
resnet20model_untouched.load_state_dict(resnet20model.state_dict())

#setting the path to store/load dataset cifar10
workdir = Path.cwd()
data_path = workdir / "datasets" / "cifar10"
if not data_path.exists():
    data_path.mkdir(parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean, std)
     ])
dataset_hparams = Hparams.DatasetHparams()
dataloaderhelper = DataLoaderHelper(0, 0, dataset_hparams)
trainset = dataloaderhelper.get_trainset(data_path, transform)
trainset, valset = dataloaderhelper.split_train_val(trainset)
trainloader = dataloaderhelper.get_train_loader(trainset)
testset = dataloaderhelper.get_testset(data_path, transform)
testloader = dataloaderhelper.get_test_loader(testset)
validationloader = dataloaderhelper.get_validation_loader(valset)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("Testing rewinding: ")
training_hparams = Hparams.TrainingHparams(dataloaderhelper.split_seed, data_order_seed=dataloaderhelper.data_order_seed,
                                           num_epoch=1, milestone_steps=[])
print("Before training:")
def compare_models(model1, model2):
    with torch.no_grad():
        list_self = Resnet_N_W.get_list_of_all_modules(model1)
        list_rewind = Resnet_N_W.get_list_of_all_modules(model2)
        for m_self, m_r in zip(list_self, list_rewind):
            if isinstance(m_self, torch.nn.Conv2d):
                conv_self = [m_self.in_channels, m_self.out_channels, m_self.kernel_size,
                            m_self.stride, m_self.padding, m_self.groups]
                conv_r = [m_r.in_channels, m_r.out_channels, m_r.kernel_size,
                            m_r.stride, m_r.padding, m_r.groups]
                if not all(x == y for x,y in zip(conv_self, conv_r)):
                    return -1
                if torch.linalg.norm(m_self.weight - m_r.weight) > 1e-10:
                    return -2
            if isinstance(m_self, torch.nn.Linear):
                linear_self = [m_self.in_features, m_self.out_features]
                linear_r = [m_r.in_features, m_r.out_features]
                if not all(x == y for x,y in zip(linear_self, linear_r)):
                    return -1
                if torch.linalg.norm(m_self.weight - m_r.weight) > 1e-10:
                    return -2
                if torch.linalg.norm(m_self.bias - m_r.bias) > 1e-10:
                    return -3
                
        return True
print("Are resnet and copy the same?", compare_models(resnet20model, resnet20model_copy))
#train a single epoch
skip = True
if not skip:
    resnet20model.to(device)
    resnet20model.train()
    optimizer = Hparams.get_optimizer(resnet20model, training_hparams)
    loss_criterion = Hparams.get_loss_criterion(training_hparams)
    print("Started training ...")
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
            print(f'[{1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            
    print("After training:")
    print("Are resnet and copy the same?", compare_models(resnet20model, resnet20model_copy))            
    resnet20model.rewind(resnet20model_copy)
    print("After rewind:")
    print("Are resnet and copy the same?", compare_models(resnet20model, resnet20model_copy))
    print("Are resnet and untouched the same?", compare_models(resnet20model, resnet20model_untouched))
    print("Are copy and untouched the same?", compare_models(resnet20model_copy, resnet20model_untouched))

#test pruning:
print("Test pruning: ")
print("Testing check_pruned: ", Resnet_N_W.check_if_pruned(resnet20model))
pruning_hparams = Hparams.PruningHparams()
resnet20model.prune(prune_ratio=pruning_hparams.pruning_ratio, method="l1")
pruned = 0
unpruned = 0
for module in Resnet_N_W.get_list_of_all_modules(resnet20model):
    if "weight_mask" in [name for name,_ in module.named_buffers()]:
        pruned += 1
    else:
        unpruned += 1
print("Total expected:", len(Resnet_N_W.get_list_of_all_modules(resnet20model)))
print("pruned:", pruned)
print("unpruned:", unpruned)
print("total: ", pruned + unpruned)
print("Testing check_pruned: ", Resnet_N_W.check_if_pruned(resnet20model))

print("Testing loaded pruned model: ")
resnet = Resnet_N_W(model_hparams)
resnet.prune(1, "identity")
resnet.load_state_dict(resnet20model.state_dict())
loaded_modules = Resnet_N_W.get_list_of_all_modules(resnet)
pruned = 0
unpruned = 0
for module in loaded_modules:
    if "weight_mask" in [name for name,_ in module.named_buffers()]:
        pruned += 1
    else:
        unpruned += 1
print("Total expected:", len(loaded_modules))
print("pruned:", pruned)
print("unpruned:", unpruned)
print("total: ", pruned + unpruned)

def calculate_sparsity(model):
    #sparsity check
    np.sum([torch.sum(module.weight == 0) for module in Resnet_N_W.get_list_of_all_modules(resnet20model)])
    + np.sum([torch.sum(module.bias == 0) for module in Resnet_N_W.get_list_of_all_modules(resnet20model)])
        #/ float(
        #    np.sum([module.weight.nelement() for module in Resnet_N_W.get_list_of_all_modules(resnet20model)])
        #    + np.sum([module.bias.nelement() for module in Resnet_N_W.get_list_of_all_modules(resnet20model)]))

#How to reproduces the a single shuffeling
generator = torch.Generator()
generator.manual_seed(0)
random_sampler = torch.utils.data.RandomSampler(data_source=trainset, generator=generator)
dataloader = torch.utils.data.DataLoader(trainset,sampler=random_sampler,
                                            batch_size=128,
                                            num_workers=1, generator=generator)
print(random_sampler.generator.get_state())

all_indices1 = list()
for inidces in dataloader.batch_sampler:
    all_indices1.append(inidces)

print(len(all_indices1))
generator.manual_seed(0)
print(random_sampler.generator.get_state())
all_indices2 = list()
for inidces in dataloader.batch_sampler:
    all_indices2.append(inidces)

diff = 0
for i, j in zip(all_indices1, all_indices2):
    if not np.isclose(np.linalg.norm(np.array(i) - np.array(j)), 0):
        diff += 1
    else:
        diff += 0

print("Diff: ", diff)
print(all_indices1[0][:10])
print(all_indices2[0][:10])


print("Testing for sequence now: ")
#How to reproduces the a training shuffeling sequence
generator = torch.Generator()
generator.manual_seed(0)
random_sampler = torch.utils.data.RandomSampler(data_source=trainset, generator=generator)
dataloader = torch.utils.data.DataLoader(trainset,sampler=random_sampler,
                                            batch_size=128,
                                            num_workers=1, generator=generator)


all_indices1 = list([list() for i in range(10)])
for epoch in range(10):
    for inidces in dataloader.batch_sampler:
        all_indices1[epoch].append(inidces)

diff_between_epochs = 0
for i, j in zip(all_indices1[0], all_indices1[1]):
    if not np.isclose(np.linalg.norm(np.array(i) - np.array(j)), 0):
        diff_between_epochs += 1
    else:
        diff_between_epochs += 0
print("Diff between epochs:", diff_between_epochs)
print(all_indices1[0][0][:10])
print(all_indices1[1][0][:10])

#generator.manual_seed(0)
random_sampler = dataloader.sampler
generator = dataloader.sampler.generator
generator.manual_seed(0)

all_indices2 = list([list() for i in range(10)])
for epoch in range(10):
    for inidces in dataloader.batch_sampler:
        all_indices2[epoch].append(inidces)

diff_between_all_epochs = 0
for epoch in range(10):
    for i, j in zip(all_indices1[epoch], all_indices2[epoch]):
        if not np.isclose(np.linalg.norm(np.array(i) - np.array(j)), 0):
            diff_between_all_epochs += 1
        else:
            diff_between_all_epochs += 0
print("Difference over all: ", diff_between_all_epochs)
print(all_indices2[0][0][:10])
print(all_indices2[1][0][:10])

#Test initialization seed
plan, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
model_hparams = Hparams.ModelHparams(plan, initializer, outputs, 0)
resnet1 = Resnet_N_W(model_hparams)
list1 = Resnet_N_W.get_list_of_all_modules(resnet1)

resnet2 = Resnet_N_W(model_hparams)
list2 = Resnet_N_W.get_list_of_all_modules(resnet2)

for module1, module2 in zip(list1, list2):
    if torch.linalg.norm(module1.weight - module2.weight) > 1e-10:
        print("Error networks are not the same")
        break
    try:
        if torch.linalg.norm(module1.bias - module2.bias) > 1e-10:
            print("Error networks are not the same")
            break
    except Exception as e:
         pass

stats = [[1, {"t": 1 }], [2, {"f": 3, "h": 5}]]

#Testing saving and loading an experiment
print("Testing save Experiment")
routines.save_experiment("e_test",
                "test",
                dataset_hparams,
                training_hparams,
                pruning_hparams,
                model_hparams,
                [resnet, resnet, resnet, resnet, resnet, resnet, resnet, resnet, resnet, resnet, resnet,],
                [stats, stats],
                override = True
)

print("Loading saved Experiment")
routines.load_experiment("e_test")

print("Testing if models stay the same during training:")
#1. Setup hyperparameters
training_hparams = Hparams.TrainingHparams(
    split_seed=dataloaderhelper.split_seed,
    data_order_seed=dataloaderhelper.data_order_seed,
    num_epoch=4)
pruning_hparams = Hparams.PruningHparams()
model_structure, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
model_hparams = Hparams.ModelHparams(
    model_structure, initializer, outputs, initialization_seed=42)
#2. Setup model
model1 = Resnet_N_W(model_hparams)
model2 = Resnet_N_W(model_hparams)
print("Are model1 and model2 the same at initalization?", compare_models(model1, model2))
early_stopper1 = EarlyStopper(model_hparams, patience=10, min_delta=0)
early_stopper2 = EarlyStopper(model_hparams, patience=10, min_delta=0)
skip2 = True
if not skip2:
    _, all_stats = routines.train(device,
        model1,
            0,
            dataloaderhelper,
            training_hparams,
            early_stopper1,
            False
            )

    routines.save_experiment(
        "test1",
        "to test if get same network, trained for 10 epochs",
        dataset_hparams, training_hparams, pruning_hparams, model_hparams,
        [model1], [all_stats],
        override = True
    )

    _, all_stats = routines.train(device,
        model2,
            0,
            dataloaderhelper,
            training_hparams,
            early_stopper2,
            False
            )

    routines.save_experiment(
        "test2",
        "to test if get same network, trained for 10 epochs",
        dataset_hparams, training_hparams, pruning_hparams, model_hparams,
        [model2], [all_stats],
        override = True
    )

    print("Are model1 and model2 the same after training?", compare_models(model1, model2))

    models1, _1, _2, _3, _4, _5 = routines.load_experiment("test1")
    m1 = models1[0]
    m1.to(device)
    print("Are model1 and loaded model1 the same?", compare_models(model1, models1[0]))
    models2, _1, _2, _3, _4, _5 = routines.load_experiment("test2")
    m2 = models2[0]
    m2.to(device)
    print("Are model2 and loaded model2 the same?", compare_models( model2, models2[0]))

print("Testing linear mode connectivity")
routines.linear_mode_connected(device, model1, model2, testloader)