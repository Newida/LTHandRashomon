from resnet20 import Resnet_N_W

class EarlyStopper:
    def __init__(self, model_hparams, patience, min_delta) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.model_hparams = model_hparams
        self.counter = 0
        self.min_val_loss = float('inf')
        self.best_model = Resnet_N_W(model_hparams)

    def __call__(self, model, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
            if not Resnet_N_W.check_if_pruned(model):
                #try loading unpruned model
                self.best_model.load_state_dict(model.state_dict())
            else:
                self.best_model.prune(1, "identity")
                self.best_model.load_state_dict(model.state_dict())
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def reset(self):
        self.counter = 0
        self.min_val_loss = float('inf')
        self.best_model = Resnet_N_W(self.model_hparams)
