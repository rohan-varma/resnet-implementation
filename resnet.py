import torch
import torch.nn as nn
import torch.nn.functional as F



def conv_3x3(in_channels, out_channels, pad=1):
	return nn.Conv2d(in_channels, out_channels, 3, padding=pad)


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		# a building block is conv-batchnorm-relu-conv-batchnorm-add-identity-relu
		super(ResidualBlock, self).__init__()
		self.conv1 = conv_3x3(in_channels, out_channels)
		self.conv2 = conv_3x3(in_channels, out_channels)


	def forward(self, x):
		identity = x
		x = F.relu(self.conv1(x))
		x = self.conv2(x)
		return F.relu(identity + x)


class Resnet(nn.Module):
	def __init__(self):
		super(Resnet, self).__init__()
		self.conv1 = conv_3x3(3, 32)
		self.conv2 = conv_3x3(32, 32)
		self.first_block = ResidualBlock(32, 32)
		self.fc_1 = nn.Linear(32**3, 64)
		self.linear = nn.Linear(64, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.first_block(x)
		x = x.view(x.shape[0], -1)
		x = F.relu(self.fc_1(x))
		x = self.linear(x)
		return x





