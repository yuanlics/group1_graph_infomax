import os.path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax

from torch_geometric.nn import GINConv, GINEConv, GENConv
from torch.nn import Sequential, Linear, ReLU, Sigmoid, PReLU, Conv1d
import torch.nn.functional as F

def main(eps,hidden_size):
  class Encoder_GIN(nn.Module):
      def __init__(self, in_channels, hidden_channels):
          super(Encoder_GIN, self).__init__()
          nn1 = Sequential(Linear(in_channels, hidden_size))
          self.conv1 = GINConv(nn1,eps=eps)
          self.prelu = nn.PReLU(hidden_channels)

      def forward(self, x, edge_index):
          x = self.conv1(x, edge_index)
          x = self.prelu(x)
          return x
  
  dataset = 'Cora'
  path = osp.join(osp.dirname(osp.realpath(sys.argv[0])), '..', 'data', dataset)
  dataset = Planetoid(path, dataset)


  class Encoder(nn.Module):
      def __init__(self, in_channels, hidden_channels):
          super(Encoder, self).__init__()
          self.conv = GCNConv(in_channels, hidden_channels, cached=True)
          self.prelu = nn.PReLU(hidden_channels)

      def forward(self, x, edge_index):
          x = self.conv(x, edge_index)
          x = self.prelu(x)
          return x


  def corruption(x, edge_index):
      return x[torch.randperm(x.size(0))], edge_index
  
  def summary(z,hidden_size):
      z = torch.cat((z.mean(dim=0), z.amax(dim=0),z.amin(dim=0)), 0)
      in_channels= z.shape[0]
      nn1 = Sequential(Linear(in_channels, hidden_size)).to(device)
      x=nn1(z)
 
      x=torch.sigmoid(x)
      
      return x


  


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # model = DeepGraphInfomax(
  #     hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
  #     summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
  #     corruption=corruption).to(device)

  model = DeepGraphInfomax(
      hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
      summary=lambda z, *args, **kwargs: torch.sigmoid(z.amin(dim=0)),
      corruption=corruption).to(device)

  # model = DeepGraphInfomax(
  #     hidden_channels=512, encoder=Encoder_GIN(dataset.num_features, 512),
  #     summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
  #     corruption=corruption).to(device)

  model = DeepGraphInfomax(
      hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
      summary=lambda z, *args, **kwargs: summary(z, 512),corruption=corruption).to(device)

  data = dataset[0].to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


  def train():
      model.train()
      optimizer.zero_grad()
      pos_z, neg_z, summary = model(data.x, data.edge_index)
      loss = model.loss(pos_z, neg_z, summary)
      loss.backward()
      optimizer.step()
      return loss.item()


  def test():
      model.eval()
      z, _, _ = model(data.x, data.edge_index)
      acc = model.test(z[data.train_mask], data.y[data.train_mask],
                      z[data.test_mask], data.y[data.test_mask], max_iter=150)
      return acc

  for epoch in range(1, 301):
      loss = train()
      #print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
  acc = test()
  # print('Accuracy: {:.4f}'.format(acc))
  return acc


if __name__ == '__main__':
    hidden_size=512
    eps_list=np.linspace(0,1,101)
    acc_all = []
    acc_list = []
    eps=0.8 #0.92(cora)
    for i in range(100):
        print('iter ', i+1)
        acc_list.append(main(eps,hidden_size))
    acc_eps = sum(acc_list) / len(acc_list)#avg in each eps
    print('Avg Accuracy: {:.4f}'.format(acc_eps))
    # for eps in eps_list:  
    #   print('eps=',eps)
    #   acc_list=[]
    #   for i in range(5):
    #     print('iter ', i+1)
    #     acc_list.append(main(eps,hidden_size))
    #   acc_eps = sum(acc_list) / len(acc_list)#avg in each eps
    #   print('Avg Accuracy: {:.4f}'.format(acc_eps))
    #   acc_all.append(acc_eps)
    # print('Max Accuracy: {:.4f}'.format(max(acc_all)))
    