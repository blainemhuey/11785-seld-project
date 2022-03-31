import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

	def _init_(self, max_pool = (5,4), out_filter=64, in_filter=64, kernel_size=3, dropout_rate=0.01):
		super().__init__()
		self.max_pool = max_pool
		self.out_filter = out_filter
		self.conv = nn.Conv2d(in_filter,out_filter, kernel_size=kernel_size)
		self.bn = nn.BatchNorm2d(out_filter)
		self.mpool = nn.MaxPool2d(self.max_pool)
		self.dropout = nn.Dropout2d(dropout_rate)

	def forward(self,x):
		x = self.conv(x)
		x = self.bn(x)
		x = nn.ReLU(x)
		x = self.mpool(x)
		x = self.dropout(x)
		return x



class Network_Seldnet(nn.Module):

	def _init_(self):

		max_pool_list = [(5,4),(1,4),(1,2)]
		self.conv_list = nn.ModuleList()
		for pool in max_pool:
			conv_list.append(
					ConvBlock(pool)
				)

		conv_out = 64*int(64/(4*4*2))
		self.rnn = nn.GRU(conv_out, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.01)
		self.rnn_act = nn.Tanh()

		self.linear = nn.Linear(128,9*14)
		self.act = nn.Tanh()

	def forward(self, x):

		for i in range(len(self.conv_list)):
			x = self.conv_list[i](x)

		x,_ = self.rnn(x)
		x = self.rnn_act(x)

		x = self.linear(x)
		x = self.act(x)

		return x




