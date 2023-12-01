import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, plan, initializer, outputs=None):
        super(Resnet_N_W, self).__init__()
        outputs = outputs or 10

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Resnet_N_W.Block(current_filters, filters, downsample)) #TODO: Resnet_N_W.Block wrong? instead just Block? why not visible
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], outputs)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize
        self.apply(initializer)

        self.initial_state = self.state_dict()

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
        """The naming scheme for a ResNet is 'resnet_N[_W]'.

        The ResNet is structured as an initial convolutional layer followed by three "segments"
        and a linear output layer. Each segment consists of D blocks. Each block is two
        convolutional layers surrounded by a residual connection. Each layer in the first segment
        has W filters, each layer in the second segment has 32W filters, and each layer in the
        third segment has 64W filters.

        The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
        N is the total number of layers in the network: 2 + 6D.
        The default value of W is 16 if it isn't provided.

        For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
        linear layer, there are 18 convolutional layers in the blocks. That means there are nine
        blocks, meaning there are three blocks per segment. Hence, D = 3.
        The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
        """
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


def kaiming_normal(w):
    if isinstance(w, nn.Linear) or isinstance(w, nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)
    
    
def kaiming_uniform(w):
    if isinstance(w, nn.Linear) or isinstance(w, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(w.weight)

def prune_model(self, model, prune_ratio=0.2, method="l1"):
    #pytroch pruning tutorials
    #TODO: implement using pytroch tutorial on pruning
    #LTH paper says: Do not prune fully-connected output layer
    #and downsampling residual connections but do i care about this?
    pass

def reinitialize_model(self, rewind_model):
    with torch.no_grad():
        for name, module in self.named_modules():
            print("name:", name)
            print("module:", module)
            module.weight.copy_(rewind_model.module)
