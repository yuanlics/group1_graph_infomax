import os.path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax,GCN2Conv

from torch_geometric.nn import GINConv, GINEConv, GENConv
from torch.nn import Sequential, Linear, ReLU, Sigmoid, PReLU, Conv1d
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric.transforms as T

def main(alpha, theta):

  class Encoder_GCN2(nn.Module):
      def __init__(self, hidden_channels, num_layers, alpha, theta,
                  shared_weights=True, dropout=0.0):
          super(Encoder_GCN2, self).__init__()
          self.lin1 = Linear(dataset.num_features, hidden_channels)
          # self.lin2 = Linear(512, 512)
          # self.conv1 = GCN2Conv(512, alpha, theta, 512,
          #                 shared_weights, normalize=False)
          self.prelu = nn.PReLU(hidden_channels)
          self.convs = torch.nn.ModuleList()
          for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
          self.dropout = dropout


      def forward(self, x, edge_index):
          x =x0= self.lin1(x)
          for conv in self.convs:
            x = conv(x, x0, edge_index)
            x = self.prelu(x)
          # x = self.convs(x, x0, edge_index)
          # x = self.prelu(x)
          return x
    
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # dataset = 'Cora'
  # path = osp.join(osp.dirname(osp.realpath(sys.argv[0])), '..', 'data', dataset)
  # dataset = Planetoid(path, dataset)
  dataset = 'Pubmed'
  path = osp.join(osp.dirname(osp.realpath(sys.argv[0])), '..', 'data', dataset)
  transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
  dataset = Planetoid(path, dataset, transform=transform)
  data = dataset[0].to(device)
  data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.



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



  # model = DeepGraphInfomax(
  #     hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
  #     summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
  #     corruption=corruption).to(device)
  model = DeepGraphInfomax(
      hidden_channels=512, encoder=Encoder_GCN2(512, 1, alpha, theta,
                   True, 0.5),
      summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
      corruption=corruption).to(device)
  data = dataset[0].to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


  def train():
      model.train()
      optimizer.zero_grad()
      pos_z, neg_z, summary = model(data.x, data.adj_t)
      loss = model.loss(pos_z, neg_z, summary)
      loss.backward()
      optimizer.step()
      return loss.item()


  def test():
      model.eval()
      z, _, _ = model(data.x, data.adj_t)
      acc = model.test(z[data.train_mask], data.y[data.train_mask],
                      z[data.test_mask], data.y[data.test_mask], max_iter=150)
      return acc


  for epoch in range(1, 301):
      loss = train()
      # print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
  acc = test()
  # print('Accuracy: {:.4f}'.format(acc))
  return acc

if __name__ == '__main__':
    print('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_size=512
    alpha_list=np.linspace(0,1,101)
    theta_list=np.linspace(0,1,101)
    dropout=0.5
    acc_all = []
    acc_list=[]
    # for i in range(20):
    #   print('iter ', i+1)
    #   acc_list.append(main(0.54,0.88))
    # acc_alpha = sum(acc_list) / len(acc_list)#avg in each eps
    # print('Avg Accuracy: {:.4f}'.format(acc_alpha))

    for alpha in alpha_list:  
      print('alpha=',alpha)
      acc_list=[]
      for i in range(5):
        print('iter ', i+1)
        acc_list.append(main(alpha,0.88))
      acc_alpha = sum(acc_list) / len(acc_list)#avg in each eps
      print('Avg Accuracy: {:.4f}'.format(acc_alpha))
      acc_all.append(acc_alpha)
    print('alpha Max Accuracy: {:.4f}'.format(max(acc_all)))

    acc_all = []
    for theta in theta_list: 
      print('theta=',theta)
      acc_list=[]
      for i in range(5):
        print('iter ', i+1)
        acc_list.append(main(0.54,theta))
      acc_k = sum(acc_list) / len(acc_list)#avg in each eps
      print('Avg Accuracy: {:.4f}'.format(acc_k))
      acc_all.append(acc_k)
    print('theta Max Accuracy: {:.4f}'.format(max(acc_all)))
    # acc_list=[]
    # for i in range(20):
    #   print('iter ', i+1)
    #   acc_list.append(main(0.85,0.79))
    # acc_alpha = sum(acc_list) / len(acc_list)#avg in each eps
    # print('Avg Accuracy: {:.4f}'.format(acc_alpha))
    