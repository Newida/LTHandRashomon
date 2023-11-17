import torch

class Hparams(object):
    """Collection of all hyperparameters used during training."""
    class DatasetHparams():
        def __init__(self, batch_size = 128) -> None:
            self.batch_size = batch_size
        #TODO: add missing:
        # transformation_seed
        # data_augmentation methods
        # otherstuff see online
    
    class ModelHparams():
        #TODO: add batchnorm_init = uniform?
        def __init__(self, model_initializer = "kaiming_normal") -> None:
            self.model_initializer = model_initializer
    
    class TrainingHparams():
        def __init__(self, optimizer_name = "sgd", lr=0.1,
                      training_steps = "160ep", data_order_seed = None,
                       momentum = 0.9, gamma = 0.1, 
                       weight_decay=1e-4, loss_criterion = "crossentropy",
                         num_epoch = 10) -> None:
            #TODO: find out what: gamma, nosterov is
            #TODO: replace num_epoch by early stopping
            #TODO: add missing hparams: 
            #training_steps, neserov_momentum, milsetone_steps, warmup_steps
            self.optimizer_name = optimizer_name
            self.lr = lr
            self.training_steps = training_steps
            self.data_order_seed = data_order_seed
            self.momentum = momentum
            self.gamma = gamma
            self.weight_decay = weight_decay
            self.loss_criterion = loss_criterion
            self.num_epoch = num_epoch
    
    class PruningHparams():
        def __init__(self, pruning_fraction = 0.2) -> None:
            self.pruning_fraction = pruning_fraction
             
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
          