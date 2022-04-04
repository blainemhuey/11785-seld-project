# import numpy as np
# import tor


class ConvBlock(nn.Module):

	def __init__(self, max_pool = (5,4), out_filter=64, in_filter=64, kernel_size=3, dropout_rate=0.01):
		super().__init__()
		self.max_pool = max_pool
		self.out_filter = out_filter
		self.conv = nn.Conv2d(in_filter,out_filter, kernel_size=kernel_size, padding=(1,1))
		self.bn = nn.BatchNorm2d(out_filter)
		self.mpool = nn.MaxPool2d(self.max_pool)
		self.dropout = nn.Dropout2d(dropout_rate)

	def forward(self,x):
		x = self.conv(x)
		x = self.bn(x)
		x = nn.ReLU()(x)
		x = self.mpool(x)
		x = self.dropout(x)
		return x



class Network_Seldnet(nn.Module):
	def __init__(self):
		super().__init__()
		# print("Here")
		max_pool_list = [(5,4),(1,4),(1,2)]
		self.conv_list = nn.ModuleList()
		for i,pool in enumerate(max_pool_list):
			# print("adding pool ")
			if i == 0:
				self.conv_list.append(
					ConvBlock(pool, 64,7)
				)
			else:
				self.conv_list.append(
					ConvBlock(pool)
				)
		print(len(self.conv_list))

		conv_out = 64*int(64/(4*4*2))
		self.rnn = nn.GRU(conv_out, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.01)
		self.rnn_act = nn.Tanh()

		self.linear = nn.Linear(128,3*config["track"] *config["classes"])
		self.linear1 = nn.Linear(128,128)
		self.act1 = nn.Tanh()
		self.act = nn.Tanh()

	def forward(self, x):
		print("Forward")
		print(len(self.conv_list))
		for i in range(len(self.conv_list)):
			x = self.conv_list[i](x)
		print(" Post conv list")
	
		x = x.transpose(1, 2).contiguous()
		x = x.view(x.shape[0], x.shape[1], -1).contiguous()
		x,_ = self.rnn(x)
		x = self.rnn_act(x)
	
		x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

		x = self.linear1(x)
		x = self.act1(x)
	
		x = self.linear(x)
		x = self.act(x)
		

		return x




