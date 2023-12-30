import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune
from utils import TorchRandomSeed

class Resnet_N_W(nn.Module):
    """Resnet_N_W as designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False):
            super(Resnet_N_W.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(f_out)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out)
                )
            else:
                self.shortcut = nn.Sequential() #empty forward

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

    def __init__(self, model_hparams):
        super(Resnet_N_W, self).__init__()

        plan = model_hparams.model_structure
        initializer = model_hparams.initializer
        weight_seed = model_hparams.initialization_seed
        outputs = model_hparams.outputs
        
        # Initial convolution
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)
        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Resnet_N_W.Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        with TorchRandomSeed(weight_seed):
            # Final fc layer. Size = number of filters in last segment.
            self.fc = nn.Linear(plan[-1][0], outputs)
            self.criterion = nn.CrossEntropyLoss()    
            # Initialize
            self.apply(initializer)

        # Helpful attributes that describe state and structure of network
        self.model_hparams = model_hparams
        self.plan = plan
        self.initializer = initializer
        self.outputs = outputs 
        self.weight_seed = weight_seed   
        self.initial_state = self.state_dict()
        self.module_list = [self.conv, self.bn, self.blocks, self.fc]

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('resnet-') and
                3 >= len(model_name.split('-')) >= 2 and #see if model_name can be parsed
                all([x.isdigit() and int(x) > 0 for x in model_name.split('-')[1:]]) and #see if containts resnet-int-int
                (int(model_name.split('-')[1]) - 2) % 6 == 0 and #valid structure of resnet
                (int(model_name.split('-')[1]) > 2)) #minimum of 2 layers required

    @staticmethod
    def is_valid_initalizer(initalizer):
        return (initalizer == "kaiming_normal") or (initalizer == "kaiming_uniform")
 
    @staticmethod
    def get_model_from_name(model_name, initializer="kaiming_normal",  outputs=10):
        if not Resnet_N_W.is_valid_initalizer(initializer):
            raise ValueError('Invalid initializer. Must be either kaiming_normal or kaiming_uniform')
        
        if initializer == "kaiming_uniform":
            initializer = kaiming_uniform
        else:
            initializer = kaiming_normal

        if not Resnet_N_W.is_valid_model_name(model_name):
            raise ValueError('Invalid ResNet model name: resnet-N-W')
        name = model_name.split('-')
        W = 16 if len(name) <= 2 else int(name[-1])
        D = int(name[1])
        if (D - 2) % 3 != 0:
            raise ValueError('Invalid ResNet depth: {}'.format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2*W, D), (4*W, D)]

        return plan, initializer, outputs
    
    @staticmethod
    def get_list_of_all_modules(model):
        all_module_list = []
        with torch.no_grad():
            for module in model.module_list:
                if isinstance(module, torch.nn.Sequential):#block
                    for name, block in module.named_children():
                        for name, module_in_block in block.named_children():
                            if isinstance(module_in_block, torch.nn.Sequential):#downsampling
                                for module_in_downsampling in module_in_block:
                                    all_module_list.append(module_in_downsampling)
                            else:
                                all_module_list.append(module_in_block)
                else:
                    all_module_list.append(module)
        return all_module_list

    @staticmethod
    def check_if_pruned(model):
        first_conv = model.module_list[0]
        if "weight_orig" in [name for name, _ in first_conv.named_parameters()]:
            return True
        else:
            return False

    @staticmethod
    def _copy_weights(source, target):
        with torch.no_grad():
            #check if pruned
            if "weight_orig" in [name for name, _ in source.named_parameters()]:
                source.weight_orig.copy_(target.weight)
                if target.bias is not None and source.bias is not None:
                    source.bias_orig.copy_(target.bias)
                elif target.bias is None and source.bias is None:
                    return
                else:
                    raise ValueError("Biases could not be matched")
            else:
                source.weight.copy_(target.weight)
                if target.bias is not None and source.bias is not None:
                    source.bias.copy_(target.bias)
                elif target.bias is None and source.bias is None:
                    return
                else:
                    raise ValueError("Biases could not be matched")

    def rewind(self, rewind_model):
        with torch.no_grad():
            list_self = Resnet_N_W.get_list_of_all_modules(self)
            list_rewind = Resnet_N_W.get_list_of_all_modules(rewind_model)
            try:
                for module_self, module_rewind in zip(list_self, list_rewind):
                    Resnet_N_W._copy_weights(module_self, module_rewind)
            except Exception as e:
                print(e)
                raise ValueError("Models do not have the same structure")
        
    def prune(self, prune_ratio, method):
        #pytroch pruning tutorials
        if method == "l1":
            prune_method = prune.L1Unstructured
        elif method == "random":
            prune_method = prune.RandomUnstructured
        elif method == "identity":
            prune_method = prune.Identity
        else:
            raise ValueError("Pruning method can only be random or l1")

        helper_list = Resnet_N_W.get_list_of_all_modules(self)
        all_parameters_to_prune = []
        for module in helper_list:
            if isinstance(module, torch.nn.Linear):
                all_parameters_to_prune.append((module, "weight"))
                all_parameters_to_prune.append((module, "bias"))
            elif isinstance(module, torch.nn.BatchNorm2d):
                all_parameters_to_prune.append((module, "weight"))
                all_parameters_to_prune.append((module, "bias"))
            else:
                all_parameters_to_prune.append((module, "weight"))

        if method == "identity":
            prune.global_unstructured(
                all_parameters_to_prune,
                pruning_method=prune_method
            )
        else:
            prune.global_unstructured(
                all_parameters_to_prune,
                pruning_method=prune_method,
                amount=prune_ratio
            )


def kaiming_normal(w):
    if isinstance(w, nn.Linear) or isinstance(w, nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)
    
    
def kaiming_uniform(w):
    if isinstance(w, nn.Linear) or isinstance(w, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(w.weight)