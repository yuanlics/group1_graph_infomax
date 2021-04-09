import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.trans = None

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        
        # should_transform = h_pl.shape[1] != h_mi.shape[1]
        # if should_transform:
        #     if self.trans is None:
        #         self.trans = nn.Linear(h_mi.shape[1], h_pl.shape[1]).cuda()
        #         self.final = nn.Linear(h_pl.shape[1], h_mi.shape[1]).cuda()
        #     h_mi = self.trans(h_mi.transpose(1, 2)).transpose(1, 2)
        # print(h_mi.shape)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
        # print(sc_1.shape)
        # if should_transform:
            # sc_1 = self.final(sc_1)
            # sc_2 = self.final(sc_2)
            # print("entered")
        # print(sc_1.shape, sc_2.shape)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

