import os.path as osp
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax

from torch_geometric.nn import GINConv, GINEConv, GENConv,APPNP
from torch.nn import Sequential, Linear, ReLU, Sigmoid, PReLU, Conv1d
import torch.nn.functional as F

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, required=True)
# parser.add_argument('--random_splits', type=bool, default=False)
# parser.add_argument('--runs', type=int, default=100)
# parser.add_argument('--epochs', type=int, default=200)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--weight_decay', type=float, default=0.0005)
# parser.add_argument('--early_stopping', type=int, default=10)
# parser.add_argument('--hidden', type=int, default=64)
# parser.add_argument('--dropout', type=float, default=0.5)
# parser.add_argument('--normalize_features', type=bool, default=True)
# parser.add_argument('--K', type=int, default=10)
# parser.add_argument('--alpha', type=float, default=0.1)
# args = parser.parse_args()

def main(alpha,k):

  class Encoder_APPNP(nn.Module):
      def __init__(self, dataset):
          super(Encoder_APPNP, self).__init__()
          self.lin1 = Linear(dataset.num_features, 512)
          # self.lin2 = Linear(512, 512)
          self.conv1 = APPNP(k, alpha)
          self.prelu = nn.PReLU(512)

      # def reset_parameters(self):
      #     self.lin1.reset_parameters()
      #     self.lin2.reset_parameters()

      # def forward(self, x, edge_index):
      #     x, edge_index = data.x, data.edge_index
      #     x = F.dropout(x, p=0.5, training=self.training)
      #     x = F.relu(self.lin1(x))
      #     x = F.dropout(x, p=0.5, training=self.training)
      #     x = self.lin2(x)
      #     x = self.prop1(x, edge_index)
      #     return F.log_softmax(x, dim=1)

      def forward(self, x, edge_index):
          x = self.lin1(x)
          x = self.conv1(x, edge_index)
          x = self.prelu(x)
          return x
    
    
  dataset = 'Citeseer'
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


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # model = DeepGraphInfomax(
  #     hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
  #     summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
  #     corruption=corruption).to(device)
  model = DeepGraphInfomax(
      hidden_channels=512, encoder=Encoder_APPNP(dataset),
      summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
      corruption=corruption).to(device)
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
    print('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_size=512
    alpha_list=np.linspace(0,1,101)
    k_list=np.linspace(0,100,11)
    dropout=0.5
    acc_all = []
    for alpha in alpha_list:  
      print('alpha=',alpha)
      acc_list=[]
      for i in range(5):
        print('iter ', i+1)
        acc_list.append(main(alpha,10))
      acc_alpha = sum(acc_list) / len(acc_list)#avg in each eps
      print('Avg Accuracy: {:.4f}'.format(acc_alpha))
      acc_all.append(acc_alpha)
    print('alpha Max Accuracy: {:.4f}'.format(max(acc_all)))
    acc_all = []
    for k in range(1,100,10):  
      print('k=',k)
      acc_list=[]
      for i in range(5):
        print('iter ', i+1)
        acc_list.append(main(0.2,k))
      acc_k = sum(acc_list) / len(acc_list)#avg in each eps
      print('Avg Accuracy: {:.4f}'.format(acc_k))
      acc_all.append(acc_k)
    print('K Max Accuracy: {:.4f}'.format(max(acc_all)))
    # acc_list=[]
    # for i in range(20):
    #   print('iter ', i+1)
    #   acc_list.append(main(0.16,10))
    # acc_alpha = sum(acc_list) / len(acc_list)#avg in each eps
    # print('Avg Accuracy: {:.4f}'.format(acc_alpha))