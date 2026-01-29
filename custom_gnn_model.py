import torch
import torch_geometric


class CustomGNN(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1=torch_geometric.nn.GATv2Conv(1, 16, heads=8, edge_dim=1, add_self_loops=False, dropout=0.25)
		self.conv2=torch_geometric.nn.GATv2Conv(16*8, 16, heads=8, edge_dim=1, add_self_loops=False, dropout=0.25)
		self.conv3=torch_geometric.nn.GATv2Conv(16*8, 16, heads=8, edge_dim=1, add_self_loops=False, dropout=0.25)
		self.conv4=torch_geometric.nn.GATv2Conv(16*8, 8, heads=8, edge_dim=1, add_self_loops=False, dropout=0.25)
		self.lin1=torch.nn.Linear(8*8, 1)
	
	def forward(self, data, edge_index):
		x=data.x
		x=self.conv1(x, edge_index, data.edge_attr)
		x=torch.nn.functional.elu(x)
		x=self.conv2(x, edge_index, data.edge_attr)
		x=torch.nn.functional.elu(x)
		x=self.conv3(x, edge_index, data.edge_attr)
		x=torch.nn.functional.elu(x)
		x=self.conv4(x, edge_index, data.edge_attr)
		return self.lin1(x)
