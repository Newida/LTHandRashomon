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
from captum.attr import IntegratedGradients, Saliency, DeepLift, NoiseTunnel
import torch.nn.functional as F
import re
import networkx as nx
import graphviz

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

dataset_hparams = Hparams.DatasetHparams(
    test_seed=0,
    val_seed=0,
    train_seed=420,
    split_seed=0,
    rngCrop_seed=0,
    rngRandomHflip_seed=0,
    batch_size=128
)

#(down)load dataset cifar10
dataloaderhelper = utils.DataLoaderHelper(
    datasethparams=dataset_hparams
)

trainset = dataloaderhelper.get_trainset(data_path, dataloaderhelper.get_transform(True))
testset = dataloaderhelper.get_testset(data_path, dataloaderhelper.get_transform(False))
trainset, valset = dataloaderhelper.split_train_val(trainset)
trainloader = dataloaderhelper.get_train_loader(trainset)
testloader = dataloaderhelper.get_test_loader(testset)
valloader = dataloaderhelper.get_validation_loader(valset)

def e1_train_val_loss(name, description):
    #initialize network
    #1. Setup hyperparameters
    training_hparams = Hparams.TrainingHparams(
        patience = 10,
        min_delta = 10,
        num_epoch = 200,
        gamma = 0.01,
        milestone_steps = [100, 150])
    pruning_hparams = Hparams.PruningHparams()
    model_structure, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
    model_hparams = Hparams.ModelHparams(
        model_structure, initializer, outputs, initialization_seed=0)
    #2. Setup model
    model = Resnet_N_W(model_hparams)
    #3. Train model
    early_stopper = EarlyStopper(
        model_hparams,
        patience=training_hparams.early_stopper_patience,
        min_delta=training_hparams.early_stopper_min_delta)
    
    _, all_stats, best_model = routines.train(device,
        model,
        0,
        dataloaderhelper,
        training_hparams,
        early_stopper,
        True
        )
    #4. Save model and statistics
    routines.save_experiment(name,
                             description,
                             dataset_hparams,
                             training_hparams,
                             pruning_hparams,
                             model_hparams,
                             [best_model],
                             [all_stats],
                             True)
    #5. Plot some results
    x_iter = []
    y_running_loss = []
    y_val_loss = []
    for stats in all_stats:
        x_iter.append(stats[0])
        stats = stats[1]
        y_running_loss.append(stats['running_loss'])
        y_val_loss.append(stats['val_loss'])

    workdir = Path.cwd()
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    saving_experiments_path = experiments_path / name
    if not saving_experiments_path.exists():
        raise ValueError("Exerpiment does not exists.")
    
    plt.plot(x_iter, y_running_loss)
    plt.yscale('log')
    plt.savefig(saving_experiments_path / "running_loss.png")
    plt.clf()
    plt.plot(x_iter, y_val_loss)
    plt.yscale('log')
    plt.savefig(saving_experiments_path / "vall_loss.png")
    return


"""start = time.time()
stats = e1_train_val_loss("e1_1", "test")
end = time.time()
print("Time of Experiment 1:", end - start)
models, all_stats, _1, _2, _3, _4 = routines.load_experiment("e1_1")
model = models[0]
model.to(device)
print("Test_acc: ", routines.get_accuracy(device, model, testloader))
print("Train_acc: ",routines.get_accuracy(device, model, trainloader))
"""

def e2_rewind_iteration(name, description, rewind_iter, init_seed, train_order_until_rewind):
    #initialize network
    #1. Setup hyperparameters
    training_hparams = Hparams.TrainingHparams(
        patience = 10,
        min_delta = 10,
        num_epoch = 200,
        gamma = 0.01,
        milestone_steps = [100, 150]
    )
    pruning_hparams = Hparams.PruningHparams(
        pruning_stopper_patience = 3,
        pruning_stopper_min_delta = 4,
        max_pruning_level = 15,
        rewind_iter = rewind_iter,
        pruning_ratio = 0.1
    )
    model_structure, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
    model_hparams = Hparams.ModelHparams(
        model_structure, initializer, outputs, initialization_seed=init_seed
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
    models, all_model_stats, best_model = routines.imp2(
        device,
        model,
        early_stopper, pruning_stopper,
        training_hparams, pruning_hparams,
        dataset_hparams=dataset_hparams,
        train_order_until_rewind=train_order_until_rewind,
        dataloaderhelper=dataloaderhelper
    )
    #4. Save model and statistics
    routines.save_experiment(
        name,
        description,
        dataset_hparams,
        training_hparams,
        pruning_hparams,
        model_hparams,
        models,
        all_model_stats,
        False
    )
    #5. Plot some results
    L_pruning_level = []
    y_test_loss = []
    for L, stats in enumerate(all_model_stats):
        L_pruning_level.append(L)
        y_test_loss.append(stats[-1][1]['test_loss'])

    workdir = Path.cwd()
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    saving_experiments_path = experiments_path / name
    if not saving_experiments_path.exists():
        raise ValueError("Exerpiment does not exists.")
    
    plt.plot(L_pruning_level[1:], y_test_loss[1:])
    #test value of untrained network makes graph harder to see
    plt.savefig(saving_experiments_path / "test_loss.png")
    return 

"""
start = time.time()
description = "rewind = 2000, initialization_seed = 123, until_rewind_seed = 0, training_seed = 420, with dataaugmentation, pruning_ratio = 0.1"
print(description)
stats = e2_rewind_iteration(
    name="e8_4", 
    description=description,
    rewind_iter=2000,
    init_seed=123,
    train_order_until_rewind=0
)
end = time.time()
print("Time of Experiment 2:", end - start)
"""

"""models, all_stats, _1, _2, _3, _4 = routines.load_experiment("e3_7")
for L, model in enumerate(models[1:]):
    model.to(device)
    print("Pruning depth: " + str(L))
    print("Density: ", Resnet_N_W.calculate_density(model))
    print("Test_acc: ", routines.get_accuracy(device, model, testloader))
    print("Train_acc: ",routines.get_accuracy(device, model, trainloader))
  """

def test_linear_mode_connectivity(name, step_size = 0.1):
    workdir = Path.cwd()
    
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    saving_experiments_path = experiments_path / name
    if not saving_experiments_path.exists():
        raise ValueError("Exerpiment does not exists.")

    models, all_stats, _1, _2, _3, _4 = routines.load_experiment(saving_experiments_path)
    all_errors = []
    #since itertools pairwise is not available
    all_errors = list()
    a, b = itertools.tee(models[1:])
    next(b, None)
    for L, (model1, model2) in enumerate(zip(a, b)):
        print("Calculating pruning depth between " + str(L) + " - " + str(L+1))
        errors = routines.linear_mode_connected(
            device,
            model1, model2,
            dataloaderhelper,
            step_size)
        if L == 0:
            all_errors += errors
        else:
            all_errors += errors[1:]
        print("Got errors of: ", errors)

    length = len(models) - 2
    x = np.linspace(0, length, int(length/step_size)+1)
    plt.clf()
    plt.xticks(np.arange(0, length+1, 1.0))
    plt.xlabel("Iteration L")
    plt.ylabel("Test Error")
    plt.plot(x, all_errors)
    plt.savefig(saving_experiments_path / "linear_mode_connectivity.png")

"""start = time.time()
test_linear_mode_connectivity("e8_4", 0.1)
end = time.time()
print("Time of linear mode connectivity:", end - start)
"""

def compare_winning_tickets(name1, name2, L, step_size = 0.1):
    workdir = Path.cwd()
    
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    saving_experiments_path1 = experiments_path / name1
    if not saving_experiments_path1.exists():
        raise ValueError("Exerpiment does not exists.")

    saving_experiments_path2 = experiments_path / name2
    if not saving_experiments_path1.exists():
        raise ValueError("Exerpiment does not exists.")

    models1, all_stats1, _1, _2, _3, _4 = routines.load_experiment(saving_experiments_path1)
    winner1 = models1[L+1] 

    models2, all_stats2, _1, _2, _3, _4 = routines.load_experiment(saving_experiments_path2)
    winner2 = models2[L+1]
    
    errors = routines.linear_mode_connected(
            device,
            winner1, winner2,
            dataloaderhelper,
            step_size)
    
    x = np.linspace(0, 1, int(1/step_size)+1)
    plt.clf()
    plt.xlabel("Interpolation points")
    plt.ylabel("Test Error")
    plt.plot(x, errors)
    plt.savefig(experiments_path / ("linear_mode_connectivity-" + name1 + "-" +
                                     name2 + "-" + str(L) + ".png"))
"""
start = time.time()
for i in [6,8]:
    compare_winning_tickets("e8_1", "e8_2", i)
    compare_winning_tickets("e8_1", "e8_3", i)
    compare_winning_tickets("e8_2", "e8_3", i)
end = time.time()
print("Time of linear mode connectivity:", end - start)
"""

def calculate_model_dissimilarity(model1, model2, dataloader, attribution_method, noise_tunnel, mode):
    #prepare models
    model1.eval()
    model2.eval()
    model1.remove_pruning()
    model2.remove_pruning()
    
    dissimilarity = 0
    normalization = 0
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    pairwise_euclid_dist = torch.nn.PairwiseDistance(p=2.0)

    if attribution_method == "grad":
        attr_algo1 = Saliency(model1)
        attr_algo2 = Saliency(model2)
        kwargs = {"abs": False}
        if noise_tunnel == True:
            attr_algo1 = NoiseTunnel(attr_algo1)
            attr_algo2 = NoiseTunnel(attr_algo2)
            kwargs = {"abs": False, "nt_type": 'smoothgrad', "nt_samples": 20, "stdevs": 0.2}

    elif attribution_method == "ig":
        attr_algo1 = IntegratedGradients(model1)
        attr_algo2 = IntegratedGradients(model2)
        #can specify baseline here but default is alread 0 so not necessary
        kwargs = {"n_steps": 100, "return_convergence_delta": False}
        if noise_tunnel == True:
            print("Cannot calculate smoothgrad for ig, since it takes to much memory")
            print("Ig without smoothgrad is performed.")
        
        #create new dataloader with smaller batch_size to avoid out of memory error
        generator = torch.Generator()
        generator.manual_seed(dataloader.generator.initial_seed())
        dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=20,
            shuffle=False,
            num_workers=1,
            generator = generator
        )
    
    else:
        print("Attribution method not known. Choose either ig or grad.")
        return
    
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        inputs.requires_grad = True
        labels = labels.to(device)
        #calculate attribution for first model
        outputs1 = model1(inputs)
        labels1 = F.softmax(outputs1, dim=1)
        prediction_score, pred_labels_idx1 = torch.topk(labels1, 1)
        pred_labels_idx1.squeeze_()
        torch.cuda.empty_cache()
        attributions1 = attr_algo1.attribute(inputs, target=pred_labels_idx1, **kwargs)
        #calculate attribution for second model
        outputs2 = model2(inputs)
        labels2 = F.softmax(outputs2, dim=1)
        prediction_score, pred_labels_idx2 = torch.topk(labels2, 1)
        pred_labels_idx2.squeeze_()
        attributions2 = attr_algo2.attribute(inputs, target=pred_labels_idx2, **kwargs)

        #calculate pairwise distances
        if mode == "positive":
            #set negative gradients to 0
            attributions1[attributions1 < 0] = 0
            attributions2[attributions2 < 0] = 0
        elif mode == "abs":
            #take the absolute values of attributions
            attributions1 = torch.abs(attributions1)
            attributions2 = torch.abs(attributions2)
        #set samples with different predictions to 0 to not count them
        prediction_diff = pred_labels_idx1 - pred_labels_idx2
        attributions1[prediction_diff != 0] = 0
        attributions2[prediction_diff != 0] = 0
        #calculate pairwise euclidian distances
        distances = pairwise_euclid_dist(attributions1, attributions2)
        dissimilarity += torch.sum(distances).item()
        normalization += dataloader.batch_size - torch.sum(prediction_diff != 0).item()

    dissimilarity/normalization

    return dissimilarity, normalization

def within_group(mode, name, iteration, save = True):
    if mode == "lossbased":
        return within_group_lossbased(name, iteration, save)
    #load models to compare
    workdir = Path.cwd()
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    winners = list()
    winner_names = list()
    pattern = re.compile((name + "_[0-9]"))
    for p in experiments_path.iterdir():
        if pattern.match(p.name):
            models, all_stats1, _1, _2, _3, _4 = routines.load_experiment(p)
            winners.append(models[iteration + 1])
            winner_names.append(name + "_" + p.name[-1])

    #reset testloader
    dataloaderhelper.reset_testloader_generator()

    #calculate pairwise comparison
    save_dict = dict()
    names_to_parameters = {"vg": ("grad", False), "sg": ("grad", True), "ig": ("ig", False)}

    for method in ["vg", "sg", "ig"]:
        print("Method:", method)
        print("-"*10)
        save_dict[method] = dict()
        for (i,j), (f_i,  f_j) in zip(itertools.combinations(winner_names, 2),
                                    itertools.combinations(winners, 2)):
            dataloaderhelper.reset_testloader_generator()
            attribution_method, nt = names_to_parameters[method]
            d, _ = calculate_model_dissimilarity(
                model1=f_i,
                model2=f_j,
                dataloader=testloader,
                attribution_method=attribution_method,
                noise_tunnel=nt,
                mode=mode
            )
            save_dict[method][i + "-" + j] = d

    if save == True:
        with open(experiments_path / (name + "_" + mode +  "_distances.pkl"), 'wb') as f:
            pickle.dump(save_dict, f)

    return save_dict

def calculate_model_dissimilarity_lossbased(model1, model2, dataloader):
    #prepare models
    model1.eval()
    model2.eval()
    model1.remove_pruning()
    model2.remove_pruning()

    #loss function of network
    loss_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    dissimilarity = 0
    normalization = 0

    for data in dataloader:
        torch.cuda.empty_cache()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        #calculate attribution for first model
        outputs1 = model1(inputs)
        labels1 = F.softmax(outputs1, dim=1)
        prediction_score, pred_labels_idx1 = torch.topk(labels1, 1)
        pred_labels_idx1.squeeze_()
        #calculate attribution for second model
        outputs2 = model2(inputs)
        labels2 = F.softmax(outputs2, dim=1)
        prediction_score, pred_labels_idx2 = torch.topk(labels2, 1)
        pred_labels_idx2.squeeze_()
        
        #set samples with same predictions to 0 to not count them
        prediction_diff = pred_labels_idx1 - pred_labels_idx2
        loss1 = loss_criterion(outputs1, labels1)
        loss2 = loss_criterion(outputs2, labels2)
        
        loss1[prediction_diff == 0] = 0
        loss2[prediction_diff == 0] = 0
        
        dissimilarity += torch.sum(torch.abs(loss1 - loss2)).item()
        normalization += dataloader.batch_size - torch.sum(prediction_diff == 0).item()

    return dissimilarity, normalization

def within_group_lossbased(name, iteration, save = True):
    #load models to compare
    workdir = Path.cwd()
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    winners = list()
    winner_names = list()
    pattern = re.compile((name + "_[0-9]"))
    for p in experiments_path.iterdir():
        if pattern.match(p.name):
            models, all_stats1, _1, _2, _3, _4 = routines.load_experiment(p)
            winners.append(models[iteration + 1])
            winner_names.append(name + "_" + p.name[-1])

    #reset testloader
    dataloaderhelper.reset_testloader_generator()

    #calculate pairwise comparison
    save_dict = dict()
    save_dict["l2_loss"] = dict()
    save_dict["classification"] = dict()

    for (i,j), (f_i,  f_j) in zip(itertools.combinations(winner_names, 2),
                                itertools.combinations(winners, 2)):
        dataloaderhelper.reset_testloader_generator()
        l2_loss_difference, classification_difference = calculate_model_dissimilarity_lossbased(
            model1=f_i,
            model2=f_j,
            dataloader=testloader
        )
        save_dict["l2_loss"][(i + "-" + j)] = l2_loss_difference
        save_dict["classification"][(i + "-" + j)] = classification_difference

    if save == True:
        with open(experiments_path / (name + "_distances_lossbased.pkl"), 'wb') as f:
            pickle.dump(save_dict, f)
    return save_dict

def plot_intra_distance_graphs(list_of_dicts, name):
    save_path = workdir / "experiments" / name
    if not save_path.exists():
        save_path.mkdir(parents=True)

    #normalize weights:
    normalizer = [1e99] * len(list_of_dicts[0])
    for i, method in enumerate(list(list_of_dicts[0].keys())):
        for j, save_dict in enumerate(list_of_dicts):
            for key, value in save_dict[method].items():
                if value < normalizer[i]:
                    normalizer[i] = value
    
    #plot distance graph
    for i, method in enumerate(list(list_of_dicts[0].keys())):
        for j, save_dict in enumerate(list_of_dicts):
            circ_graph = create_graphviz_description_intra(save_dict, method, normalizer[i])
            graph = graphviz.Source(circ_graph, engine="circo")
            node1, node2 = [node.split("_")[0] for node in list(save_dict[method].keys())[0].split("-")]
            graph.render(save_path / (method + "_" + node1), format='png', cleanup=True)

def create_graphviz_description_intra(save_dict, method, normalizer):
    s = '''
    graph bipartite {
        edge [style="dashed"]

        node [shape=circle];

    '''

    all_labels = list()
    for key, value in save_dict[method].items():
        label = round(value/normalizer, 2)
        all_labels.append(label)
    
    color_map = define_map_labels_to_colors(all_labels)

    for n1, n2 in [node.split("-") for node in list(save_dict[method].keys())]:
        s += n1 + "; "
        s += n2 + "; "

    s += "\n"
    for edge, value in save_dict[method].items():
        s += (" -- ").join(edge.split("-"))
        label = round(value/normalizer, 2)
        s += "[label=" + str(label) + ";"
        s += "fontcolor=" + str(color_map[label]) + ", color=" + str(color_map[label])
        s+= "];\n"

    s += "}"
    return s

def visualize_results_intra(mode):
    workdir = Path.cwd()
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    if mode == "lossbased":
        pattern = re.compile(("e[0-9]_distances_lossbased.pkl"))
    else:
        pattern = re.compile(("e[0-9]_" + mode + "_distances.pkl"))

    list_of_dicts = list()
    for p in experiments_path.iterdir():
        if pattern.match(p.name):
            with open(p, "rb") as f:
                loaded_dict = pickle.load(f)
            list_of_dicts.append(loaded_dict)

    plot_intra_distance_graphs(list_of_dicts, mode + "intraResults")


def between_groups(name1, name2, mode, iteration, save = True):

    if mode == "lossbased":
        return between_groups_lossbased(name1, name2, iteration, save)
    #load models to compare
    workdir = Path.cwd()
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    winners1 = list()
    winner_names1 = list()
    pattern1 = re.compile((name1 + "_[0-9]"))
    winners2 = list()
    winner_names2 = list()
    pattern2 = re.compile((name2 + "_[0-9]"))
    for p in experiments_path.iterdir():
        if pattern1.match(p.name):
            models, all_stats1, _1, _2, _3, _4 = routines.load_experiment(p)
            winners1.append(models[iteration + 1])
            winner_names1.append(name1 + "_" + p.name[-1])
        elif pattern2.match(p.name):
            models, all_stats1, _1, _2, _3, _4 = routines.load_experiment(p)
            winners2.append(models[iteration + 1])
            winner_names2.append(name2 + "_" + p.name[-1])

    #reset testloader
    dataloaderhelper.reset_testloader_generator()

    #calculate pairwise comparison
    save_dict = dict()
    names_to_parameters = {"vg": ("grad", False), "sg": ("grad", True), "ig": ("ig", False)}

    for method in ["vg", "sg", "ig"]:
        print("Method:", method)
        print("-"*10)
        save_dict[method] = dict()
        for (i,j), (f_i,  f_j) in zip(itertools.product(winner_names1, winner_names2),
                                    itertools.product(winners1, winners2)):
            dataloaderhelper.reset_testloader_generator()
            attribution_method, nt = names_to_parameters[method]
            d, _ = calculate_model_dissimilarity(
                model1=f_i,
                model2=f_j,
                dataloader=testloader,
                attribution_method=attribution_method,
                noise_tunnel=nt,
                mode=mode
            )
            save_dict[method][(i + "-" + j)] = d

    if save == True:
        with open(experiments_path / (name1 + "_" + name2 + "_" + mode +  "_distances.pkl"), 'wb') as f:
            pickle.dump(save_dict, f)

    return save_dict

def between_groups_lossbased(name1, name2, iteration, save = True):
    #load models to compare
    workdir = Path.cwd()
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    winners1 = list()
    winner_names1 = list()
    pattern1 = re.compile((name1 + "_[0-9]"))
    winners2 = list()
    winner_names2 = list()
    pattern2 = re.compile((name2 + "_[0-9]"))
    for p in experiments_path.iterdir():
        if pattern1.match(p.name):
            models, all_stats1, _1, _2, _3, _4 = routines.load_experiment(p)
            winners1.append(models[iteration + 1])
            winner_names1.append(name1 + "_" + p.name[-1])
        elif pattern2.match(p.name):
            models, all_stats1, _1, _2, _3, _4 = routines.load_experiment(p)
            winners2.append(models[iteration + 1])
            winner_names2.append(name2 + "_" + p.name[-1])

    #reset testloader
    dataloaderhelper.reset_testloader_generator()

    #calculate pairwise comparison
    save_dict = dict()
    save_dict["l2_loss"] = dict()
    save_dict["classification"] = dict()

    for (i,j), (f_i,  f_j) in zip(itertools.product(winner_names1, winner_names2),
                                itertools.product(winners1, winners2)):
        dataloaderhelper.reset_testloader_generator()
        l2_loss_difference, classification_difference = calculate_model_dissimilarity_lossbased(
            model1=f_i,
            model2=f_j,
            dataloader=testloader
        )
        save_dict["l2_loss"][(i + "-" + j)] = l2_loss_difference
        save_dict["classification"][(i + "-" + j)] = classification_difference

    if save == True:
        with open(experiments_path / (name1 + "_" + name2 + "_distances_lossbased.pkl"), 'wb') as f:
            pickle.dump(save_dict, f)

    return save_dict


def visualize_distances_inter_and_intra(mode):
    workdir = Path.cwd()
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    if mode == "lossbased":
        pattern = re.compile(("e[0-9]_e[0-9]_distances_lossbased.pkl"))
    else:
        pattern = re.compile(("e[0-9]_e[0-9]_" + mode + "_distances.pkl"))

    list_of_dicts_inter = list()
    for p in sorted(experiments_path.iterdir()):
        if pattern.match(p.name):
            with open(p, "rb") as f:
                loaded_dict = pickle.load(f)
            list_of_dicts_inter.append(loaded_dict)

    if mode == "lossbased":
        pattern = re.compile(("e[0-9]_distances_lossbased.pkl"))
    else:
        pattern = re.compile(("e[0-9]_" + mode + "_distances.pkl"))
    
    #get intras results for calculating the normalizer
    list_of_dicts_intra = list()
    for p in sorted(experiments_path.iterdir()):
        if pattern.match(p.name):
            with open(p, "rb") as f:
                loaded_dict = pickle.load(f)
            list_of_dicts_intra.append(loaded_dict)

    #find normalization weights per xml method
    normalizer = [1e99] * len(list_of_dicts_intra[0])
    for i, method in enumerate(list(list_of_dicts_intra[0].keys())):
        for j, save_dict in enumerate(list_of_dicts_intra):
            for key, value in save_dict[method].items():
                if value < normalizer[i]:
                    normalizer[i] = value

    
    plot_inter_distance_graphs(list_of_dicts_inter, normalizer, mode + "Results")

def plot_inter_distance_graphs(list_of_dicts_inter, normalizer, name):
    save_path = workdir / "experiments" / name
    if not save_path.exists():
        save_path.mkdir(parents=True)
    
    #plot distance graph
    for i, method in enumerate(list(list_of_dicts_inter[0].keys())):
        for j, save_dict in enumerate(list_of_dicts_inter):
            dot_graph = create_graphviz_description(save_dict, method, normalizer[i])
            graph = graphviz.Source(dot_graph)
            node1, node2 = [node.split("_")[0] for node in list(save_dict['vg'].keys())[0].split("-")]
            graph.render(save_path / (method + "_" + node1 + "_" + node2), format='png', cleanup=True)

def define_map_labels_to_colors(values):
    min_value = min(values)
    max_value = max(values)

    min_integer = 5
    max_integer = 10

    color_map = dict()
    for value in values:
        color = min_integer + int(((value - min_value) / (max_value - min_value)) * (max_integer - min_integer))
        color_map[value] = color
    
    return color_map

def create_graphviz_description(save_dict, method, normalizer):
    s = '''
    graph bipartite {
        edge [style="dashed", colorscheme=Greens9]

    '''
    l1 = "{rank=same;"
    l2 = "{rank=same;"

    all_labels = list()
    for key, value in save_dict[method].items():
        label = round(value/normalizer, 2)
        all_labels.append(label)
    
    color_map = define_map_labels_to_colors(all_labels)

    for key, value in save_dict[method].items():
        node1, node2 = key.split("-")
        label = round(value/normalizer, 2)
        s += "\t" + node1 + " -- " + node2 + " [label=" + str(label) + ", "
        s += "fontcolor=" + str(color_map[label]) + ", color=" + str(color_map[label])
        s += "]\n"
        l1 += node1 + ";"
        l2 += node2 + ";"

    s += "\n"
    s += l1 + "}" + "\n"
    s += l2 + "}" + "\n"
    s += "}"
    return s

"""
start = time.time()
save_dict = within_group("abs", "e6", 8)
end = time.time()
print("Time of comparison:", end - start)
"""
visualize_results_intra("lossbased")

#start = time.time()
#save_dict = between_groups("e6", "e7", "lossbased", 8, save = True)
#end = time.time()
#print("Time of comparison:", end - start)
visualize_distances_inter_and_intra("positive")