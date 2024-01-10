import pickle
import torch
import time
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torchvision
import torchvision.transforms as transforms
from resnet20 import Resnet_N_W
from Hparams import Hparams
from utils_Earlystopper import EarlyStopper
import utils
import routines
import copy

try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except NameError or ModuleNotFoundError:
    pass

#setting the path to store/load dataset cifar10
workdir = Path.cwd()
data_path = workdir / "datasets" / "cifar10"
if not data_path.exists():
    data_path.mkdir(parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_state = 0 #TODO: add to commandline arguments later

with utils.TorchRandomSeed(random_state):
#data augmentation
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    dataset_hparams = Hparams.DatasetHparams()
    #(down)load dataset cifar10
    dataloaderhelper = utils.DataLoaderHelper(
        split_seed=0,
        data_order_seed=0,
        val_seed=0,
        test_seed=0,
        datasethparams=dataset_hparams
        )
    trainset = dataloaderhelper.get_trainset(data_path, transform)
    testset = dataloaderhelper.get_testset(data_path, transform)
    trainset, valset = dataloaderhelper.split_train_val(trainset)
    trainloader = dataloaderhelper.get_train_loader(trainset)
    testloader = dataloaderhelper.get_test_loader(testset)
    valloader = dataloaderhelper.get_validation_loader(valset)

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
                d = torch.linalg.norm(m_self.weight.cpu() - m_r.weight.cpu())
                if torch.linalg.norm(m_self.weight.cpu() - m_r.weight.cpu()) > 1e-16:
                    return -2
            if isinstance(m_self, torch.nn.Linear):
                linear_self = [m_self.in_features, m_self.out_features]
                linear_r = [m_r.in_features, m_r.out_features]
                if not all(x == y for x,y in zip(linear_self, linear_r)):
                    return -1
                if torch.linalg.norm(m_self.weight.cpu() - m_r.weight.cpu()) > 1e-16:
                    return -2
                if torch.linalg.norm(m_self.bias.cpu() - m_r.bias.cpu()) > 1e-16:
                    return -3
                
        return True
    
def test_convex_weights(beta, convex_model, model1, model2):
    with torch.no_grad():
        list_model1 = Resnet_N_W.get_list_of_all_modules(model1)
        list_model2 = Resnet_N_W.get_list_of_all_modules(model2)
        list_convex_model = Resnet_N_W.get_list_of_all_modules(convex_model)
        for m_convex, m1, m2 in zip(list_convex_model, list_model1, list_model2):
            if isinstance(m1, torch.nn.Conv2d):
                if torch.linalg.norm(m_convex.weight.cpu() -
                    ((1 - beta) * m1.weight.cpu() + beta * m2.weight.cpu())) > 1e-16:
                    return -2
            if isinstance(m1, torch.nn.Linear):
                if torch.linalg.norm(m_convex.weight.cpu() -
                    ((1 - beta) * m1.weight.cpu() + beta * m2.weight.cpu())) > 1e-16:
                    return -2
                if torch.linalg.norm(m_convex.bias.cpu() -
                    ((1 - beta) * m1.bias.cpu() + beta * m2.bias.cpu())) > 1e-16:
                    return -3
                
        return True


training_hparams = Hparams.TrainingHparams(
        split_seed=dataloaderhelper.split_seed,
        data_order_seed=dataloaderhelper.data_order_seed,
        patience = 10,
        min_delta = 4,
        num_epoch = 1,
        gamma = 0.01,
        milestone_steps = [100, 150])
pruning_hparams = Hparams.PruningHparams(
        pruning_stopper_patience = 3,
        pruning_stopper_min_delta = 4,
        max_pruning_level = 3,
        rewind_iter = 0,
        pruning_ratio = 0.2
    )
model_structure, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
model_hparams = Hparams.ModelHparams(
        model_structure, initializer, outputs, initialization_seed=0
        )
    #2. Setup model
model = Resnet_N_W(model_hparams)
    #3. Train model
early_stopper = EarlyStopper(
        model_hparams,
        patience=training_hparams.early_stopper_patience,
        min_delta=training_hparams.early_stopper_min_delta
        )
pruning_stopper = EarlyStopper(
        model_hparams,
        patience=pruning_hparams.pruning_stopper_patience,
        min_delta=pruning_hparams.pruning_stopper_min_delta
        )
    
loss_criterion = Hparams.get_loss_criterion(training_hparams)
max_pruning_level = pruning_hparams.max_pruning_level
rewind_iter = pruning_hparams.rewind_iter
models = []
all_model_stats = []
#create copy of initial network and save it:
save_model = Resnet_N_W(model.model_hparams)
save_model.load_state_dict(model.state_dict())
models.append(save_model)

dataloaderhelper.reset_testoader_generator()
test_loss = routines.get_loss(device, model, dataloaderhelper.testloader, loss_criterion)
print('|' + str(-1) + '| test_loss: ' + str(test_loss))
    
all_model_stats.append([[-1, {"test_loss": test_loss}]]) #initial model doesnt need stats calculated
rewind_point = None    
for L in range(0, max_pruning_level):
    #do training
    rewind_model, all_stats, best_model = routines.train(
        device,
        model, rewind_iter,
        dataloaderhelper, training_hparams,
        early_stopper,
        False
    )
    early_stopper.reset()
    if rewind_point is None:
        rewind_point = rewind_model
    #pruning
    best_model.prune(
        prune_ratio = pruning_hparams.pruning_ratio,
        method = pruning_hparams.pruning_method
    )
    print("Density: ", Resnet_N_W.calculate_density(best_model))
    #create copy of found network and save it:
    save_model = best_model.copy()
    models.append(save_model)

    print("Expected: Yes", compare_models(best_model, models[-1]))
    
    #test if early stop
    dataloaderhelper.reset_testoader_generator()
    test_loss = routines.get_loss(device, best_model, dataloaderhelper.testloader, loss_criterion)
    print('|' + str(L) + '| test_loss: ' + str(test_loss))
    all_stats.append([-1, {"test_loss" : test_loss}])
    #save statistics calculated during training
    all_model_stats.append(all_stats)
    
    if pruning_stopper(best_model, test_loss):
        print("Pruning: Stopped early")
        print("Trained for " + str(L) + " Pruning-Iterations.")
        print("Got minimum test loss in early stopper: ", pruning_stopper.min_val_loss)
    
    #rewind model
    best_model.rewind(rewind_point)
    print("Density of model:", Resnet_N_W.calculate_density(best_model))

def linear_mode_connected(device, model1, model2, dataloaderhelper, betas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
    with torch.no_grad():
        testloader = dataloaderhelper.testloader
       
        if not Resnet_N_W.check_if_pruned(model1):
            model1.prune(1, "identity")
        if not Resnet_N_W.check_if_pruned(model1):
            model2.prune(1, "identity")
        model2.to(device)
        list_model2 = Resnet_N_W.get_list_of_all_modules(model2)
        list_model1 = Resnet_N_W.get_list_of_all_modules(model1)
        errors = []

        for beta in betas:
            convex_network = model1.copy()
            list_convex_network = Resnet_N_W.get_list_of_all_modules(convex_network)
            for module in list_convex_network:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.utils.prune.remove(module, 'weight')
                    torch.nn.utils.prune.remove(module, 'bias')
                elif isinstance(module, torch.nn.BatchNorm2d):
                    torch.nn.utils.prune.remove(module, 'weight')
                    torch.nn.utils.prune.remove(module, 'bias')
                else:
                    torch.nn.utils.prune.remove(module, 'weight')
                
            print("Are models the same: Expected Yes", compare_models(model1, convex_network))
            list_convex_network = Resnet_N_W.get_list_of_all_modules(convex_network)
            for conv, m1, m2 in zip(list_convex_network, list_model1, list_model2):
                convex_weigths(conv, m1, m2, beta)
            print("Test calculation: Expected Yes", test_convex_weights(beta, convex_network, model1, model2))
            dataloaderhelper.reset_testoader_generator()
            errors.append(1 - routines.get_accuracy(device, convex_network, testloader))
            
        return errors
    
def convex_weigths(conv, m1, m2, beta):
    with torch.no_grad():
        conv.weight.copy_((1 - beta) * m1.weight + beta * m2.weight)
        if m1.bias is not None and m2.bias is not None:
            conv.bias.copy_((1 - beta) * m1.bias + beta * m2.bias)
        elif m1.bias is None and m2.bias is None:
            return
        else:
            raise ValueError("Biases could not be matched")

all_errors = list()
a, b = itertools.tee(models[1:])
next(b, None)
for L, (model1, model2) in enumerate(zip(a, b)):
    print("Testing pruning depth: " + str(L))
    errors = linear_mode_connected(
        device,
        model1, model2,
        dataloaderhelper)
    if L == 0:
        all_errors += errors
    else:
        print("Check errors: Expected close to 0:", all_errors[-1] - errors[0])
        all_errors += errors[1:]
    print("Got errors of: ", errors)

length = len(models) - 2
print("x:", np.linspace(0, length, num=11*length-1))
print("y:", all_errors)
plt.plot(np.linspace(0, length, num=11*length-1), all_errors) #11 since len(beta) = 11
plt.savefig("linear_mode_connectivity.png")