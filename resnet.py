import torch
import torch.nn as nn
import torch.nn.functional as F



def conv_3x3(in_channels, out_channels, pad=1):
	return nn.Conv2d(in_channels, out_channels, 3, padding=pad)


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
		self.conv2 = conv_3x3(in_channels, out_channels)

	def forward(self, x):
		identity = x
		x = F.relu(self.conv1(x))
		x = self.conv2(x)
		if x.shape != identity.shape:
			print(f'Shape mismatch - x: {x.shape}, original identity: {identity.shape}')
			identity = F.pad(identity, pad=(0,0,0,0,8,8)) # TODO - instead of 8, compute difference in dimensionality and pad based on that (half each), this only currently works if the dim is 16. Also fix the docs on https://pytorch.org/docs/master/_modules/torch/nn/functional.html#pad!!
		print(f': {x.shape}, {identity.shape}')
		return F.relu(identity + x)

	def __repr__(self):
		return f'Residual block with in_channels {self.in_channels} and out channels {self.out_channels}'


class Resnet(nn.Module):
	def __init__(self, n=1):
		super(Resnet, self).__init__()
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
		for b in self.residual_blocks:
			print(b)

		self.linear = nn.Linear(32**3, 10)

		self.conv1 = conv_3x3(3, 16)
		# self.conv2 = conv_3x3(32, 32)
		# self.first_block = ResidualBlock(32, 32)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		for block in self.residual_blocks:
			x = block(x)
		# x = self.first_block(x)
		x = x.view(x.shape[0], -1)
		x = self.linear(x)
		return x





