import torch
import torch.nn as nn
from layers import GAT, GCN, AvgReadout, Discriminator, Discriminator2
import pdb

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.disc2 = Discriminator2(n_h)

    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2, aug_type):
        
        h_0 = self.gcn(seq1, adj, sparse)
        h_2 = self.gcn(seq2, adj, sparse)
        if aug_type == 'edge':

            h_1 = self.gcn(seq1, aug_adj1, sparse)
            h_3 = self.gcn(seq1, aug_adj2, sparse)

        elif aug_type == 'mask':

            h_1 = self.gcn(seq3, adj, sparse)
            h_3 = self.gcn(seq4, adj, sparse)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = self.gcn(seq3, aug_adj1, sparse)
            h_3 = self.gcn(seq4, aug_adj2, sparse)
            
        else:
            assert False
            
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_3, msk)
        c_3 = self.sigm(c_3)

        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        c_0 = self.read(h_0, msk)
        c_0 = self.sigm(c_0)

        h_2 = self.gcn(seq2, adj, sparse)
        # print(c_1.shape, c_3.shape, h_0.shape, h_2.shape)
        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)

        # print(c_0.shape, h_1.shape, h_2.shape, h_3.shape)
        h_1corrupt = h_1.squeeze()
        h_1corrupt = h_1corrupt[torch.randperm(h_1corrupt.size(0))].unsqueeze(0)

        h_3corrupt = h_3.squeeze()
        h_3corrupt = h_3corrupt[torch.randperm(h_3corrupt.size(0))].unsqueeze(0)

        ret3 = self.disc(c_0, h_1, h_3corrupt, samp_bias1, samp_bias2)
        ret4 = self.disc(c_0, h_3, h_1corrupt, samp_bias1, samp_bias2)

        ret_a = ret1 + ret2
        ret_b = ret3 + ret4
        return ret_a, ret_b

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

