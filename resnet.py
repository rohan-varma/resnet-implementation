import torch
import torch.nn as nn
import torch.nn.functional as F


debug = True

# TODO stride=2 is used in the paper but this is causing shape issues right now :(
def conv_3x3(in_channels, out_channels, pad=1, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=pad)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A ResidualBlock implements the basic residual block discussed in the paper, for the network used on CIFAR 10.
        It consists of a pair of 3x3 convolutional layers, all with the same output feature map size (either 32, 16, or 8)
        We apply conv-BN-relu for the first layer, then conv-BN, then add the input (residual connection) and do a final RELU.
        We zero-pad the input for dimension mismatches, so that no new parameters are introduced in the residual connections.
        """
        self.in_channels, self.out_channels = in_channels, out_channels
        super(ResidualBlock, self).__init__()
        self.conv1 = conv_3x3(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv2 = conv_3x3(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # The paper said that identity shortcuts are used in all cases, 
        # so in cases of shape mismatch I pad to align dimensions, which introduces no new parameters.
        if x.shape != identity.shape:
            shape_diff = abs((sum(identity.shape) - sum(x.shape)))
            identity = F.pad(identity, pad=(0,0,0,0,shape_diff//2,shape_diff//2))
        x = F.relu(identity + x)
        return x

    def __repr__(self):
        return f'Residual block with in_channels {self.in_channels} and out channels {self.out_channels}'


class Resnet(nn.Module):
    def __init__(self, n=1, dbg=False):
        super(Resnet, self).__init__()
        debug = dbg
        self.residual_blocks = []
        # create number of residual blocks needed
        cur_feature_map_size = 16
        changed = False
        for i in range(3*n):
            if i != 0 and i % n == 0:
                cur_feature_map_size = cur_feature_map_size*2
                changed = True
            block = ResidualBlock(cur_feature_map_size if not changed else cur_feature_map_size//2, cur_feature_map_size)
            changed = False
            self.residual_blocks.append(block)

        self.linear = nn.Linear(16384, 10)
        self.conv1 = conv_3x3(3, 16)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        for block in self.residual_blocks:
            x = block(x)
        # x = self.first_block(x)
        # flatten the multidimensional input to a single matix for input into the FC layer
        x = self.pool(x) # only difference is this pool
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    resnet = Resnet()