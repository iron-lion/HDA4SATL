import torch.nn as nn
from torch.nn import functional as F
import torch


class EncoderBlk(nn.Module):
    def __init__(self, input_len, output_len, bn=False, relu=0, dr=0):
        super(EncoderBlk, self).__init__()
        layers = [nn.Linear(in_features = input_len, out_features = output_len)]
        if bn:
            layers.append(nn.BatchNorm1d(num_features=output_len))
        if relu > 0:
            layers.append(nn.ReLU(relu))
        if dr > 0:
            layers.append(nn.Dropout(dr))
        
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)


class DecoderBlk(nn.Module):
    def __init__(self, input_len, output_len, bn=False, relu=0, dr=0):
        super(DecoderBlk, self).__init__()
        layers = [nn.Linear(in_features = input_len, out_features = output_len)]
        if bn:
            layers.append(nn.BatchNorm1d(num_features=output_len))
        if relu > 0:
            layers.append(nn.ReLU(relu))
        if dr > 0:
            layers.append(nn.Dropout(dr))
        
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)



class VGEPEncoder(nn.Module):
    def __init__(self, channel_len : int, layers : list, latent_size : int, bn=True, relu=0.1, device=None):
        super(VGEPEncoder, self).__init__()
        l = [channel_len]
        l += layers

        this_len = l[0]

        self.blks = nn.ModuleList()
        for l_len in l[1:]:
            self.blks.append(EncoderBlk(this_len, l_len, bn, relu).to(device))
            this_len = l_len
        
        self._mu = nn.Sequential(
                    #nn.BatchNorm1d(num_features = this_len),
                    nn.Linear(in_features = this_len, out_features = latent_size),
                    )
        self._logvar = nn.Sequential(
                    #nn.BatchNorm1d(num_features = this_len),
                    nn.Linear(in_features = this_len, out_features = latent_size)
                    )

    def forward(self,x):
        layer1 = self.blks[0]
        out = layer1(x)
        for blk in self.blks[1:]:
            out = blk(out)
       
        mu = self._mu(out)
        logvar = torch.softmax(self._logvar(out), dim=1)
        return mu, logvar


class AEncoder(nn.Module):
    def __init__(self, channel_len, layers, latent_size, bn=True, relu=0.1, device=None):
        super(AEncoder, self).__init__()
        l = [channel_len]
        l += layers
        l.append(latent_size)

        this_len = l[0]

        self.blks = nn.ModuleList()
        for l_len in l[1:]:
            self.blks.append(EncoderBlk(this_len, l_len, bn, relu).to(device))
            this_len = l_len
        

    def forward(self,x):
        layer1 = self.blks[0]
        out = layer1(x)
        for blk in self.blks[1:]:
            out = blk(out)
        
        return out


class ADecoder(nn.Module):
    def __init__(self, channel_len, layers, latent_size, bn=True, relu=0.1, device=None):
        super(ADecoder, self).__init__()
        l = [latent_size]
        l += layers
        l.append(channel_len)

        this_len = l[0]

        self.blks = nn.ModuleList()
        for l_len in l[1:]:
            self.blks.append(EncoderBlk(this_len, l_len, bn, relu).to(device))
            this_len = l_len
        

    def forward(self,x):
        layer1 = self.blks[0]
        out = layer1(x)
        for blk in self.blks[1:]:
            out = blk(out)
        
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.5)
        m.weight.data.zero_()
        m.bias.data = torch.ones(m.bias.data.size())
