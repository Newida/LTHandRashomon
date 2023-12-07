import torch
import numpy as np
from resnet20 import Resnet_N_W
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from Hparams import Hparams
from utils_DataLoader import DataLoaderHelper

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
resnet20model = Resnet_N_W(plan, initializer, outputs)
print(resnet20model.blocks)
print("-Testing forwardstep")
input = torch.unsqueeze(torch.stack([torch.eye(32),torch.eye(32),torch.eye(32)]), 0)
resnet20model(input)
print("-"*20)

#Test reintialization
plan, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
resnet20model = Resnet_N_W(plan, initializer, outputs)
resnet20model_copy = Resnet_N_W(plan, initializer, outputs)
resnet20model_copy.load_state_dict(resnet20model.state_dict())
resnet20model_untouched = Resnet_N_W(plan, initializer, outputs)
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
trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=dataset_hparams.batch_size,
                                         shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("Testing rewinding: ")
training_hparams = Hparams.TrainingHparams(num_epoch=1, milestone_steps=[])
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

print("Testing loaded pruned model: ")
resnet = Resnet_N_W(plan, initializer, outputs)
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

#sparsity check fix this and include it into calculate_stats
print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            np.sum([torch.sum(module.weight == 0) for module in Resnet_N_W.get_list_of_all_modules(resnet20model)])
            + np.sum([torch.sum(module.bias == 0) for module in Resnet_N_W.get_list_of_all_modules(resnet20model)]))
        / float(
            np.sum([module.weight.nelement() for module in Resnet_N_W.get_list_of_all_modules(resnet20model)])
            + np.sum([module.bias.nelement() for module in Resnet_N_W.get_list_of_all_modules(resnet20model)]))
        )
    )

unpruned = 0
pruned = 0
for module in Resnet_N_W.get_list_of_all_modules(resnet20model):
    for name, _ in list(module.named_buffers()):
        if not "weight_mask" == name:
            unpruned += 1
        else:
            pruned += 1
print("pruned:", pruned)
print("unpruned:", unpruned)