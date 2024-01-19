import torch

class Hparams(object):
    """Collection of all hyperparameters used during training."""
    class DatasetHparams():
        def __init__(self, test_seed, val_seed, train_seed, split_seed, rngCrop = None, rngRandomHflip = None, batch_size = 128) -> None:
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.247, 0.243, 0.261]
            self.rngCrop = rngCrop
            self.rngRandomHflip = rngRandomHflip
            self.batch_size = batch_size
            self.test_seed = test_seed
            self.val_seed = val_seed
            self.split_seed = split_seed
            self.train_seed = train_seed     

    class ModelHparams():
        def __init__(self, model_structure, initializer, outputs, initialization_seed = 0) -> None:
            self.initialization_seed = initialization_seed
            self.model_structure = model_structure
            self.initializer = initializer
            self.outputs = outputs
    
    class TrainingHparams():
        def __init__(self, split_seed, data_order_seed, optimizer_name = "sgd", lr=0.1,
                      momentum = 0.9, gamma = 0.1, 
                      weight_decay=1e-4, loss_criterion = "crossentropy",
                      num_epoch = 160,
                      milestone_steps = [80, 120],
                      patience = 10,
                      min_delta = 4) -> None:
            
            self.optimizer_name = optimizer_name
            self.lr = lr
            self.data_order_seed = data_order_seed
            self.split_seed = split_seed
            self.momentum = momentum
            self.weight_decay = weight_decay
            self.loss_criterion = loss_criterion
            self.num_epoch = num_epoch
            self.milestone_steps = milestone_steps #at which epoch to drop the learning rate
            #measured in epochs
            self.gamma = gamma #the amount the learning rate drops when reaching a milestone
            self.early_stopper_patience = patience
            self.early_stopper_min_delta = min_delta

    class PruningHparams():
        def __init__(self,
                    max_pruning_level = 12,
                    rewind_iter = 0,
                    pruning_ratio = 0.2,
                    method = "l1",
                    pruning_stopper_patience = 3,
                    pruning_stopper_min_delta = 0) -> None:
            self.pruning_ratio = pruning_ratio
            self.pruning_method = method
            self.max_pruning_level = max_pruning_level
            self.rewind_iter = rewind_iter
            self.pruning_stopper_patience = pruning_stopper_patience
            self.pruning_stopper_min_delta = pruning_stopper_min_delta
             
    @staticmethod
    def get_optimizer(model, trainingHparams):
               if trainingHparams.optimizer_name == "sgd":
                    return torch.optim.SGD(
                         model.parameters(),
                         lr = trainingHparams.lr,
                         momentum = trainingHparams.momentum,
                         weight_decay = trainingHparams.weight_decay or 0
                    )
               if trainingHparams.optimizer_name == "adam":
                    return torch.optim.Adam(
                         model.parameters(),
                         lr = trainingHparams.lr,
                         weight_decay = trainingHparams.weight_decay or 0
                    )
               else:
                    raise ValueError("No such optimizer: choose either sgd or adam")
               
    @staticmethod
    def get_loss_criterion(trainingHparams):
          if trainingHparams.loss_criterion == "crossentropy":
               return torch.nn.CrossEntropyLoss()
          if trainingHparams.loss_criterion == "mse":
               return torch.nn.MSELoss()
          else:
               raise ValueError("No such loss: choose either crossentropy or mse")
          
    @staticmethod
    def get_lr_scheduler(optimizer, trainingHparams):
         return torch.optim.lr_scheduler.MultiStepLR(
               optimizer,
               milestones=trainingHparams.milestone_steps,
               gamma=trainingHparams.gamma
               )
