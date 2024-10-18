import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=10):
        super(Proxy_Anchor, self).__init__()
        # Proxy Anchor Initialization
        self.proxies = torch.randn(nb_classes, sz_embed).cuda()
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        # Calculate cosine similarity with L2 normalization
        cos = F.linear(l2_norm(X), l2_norm(P))
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        # Clamp values to prevent exponential overflow
        pos_exp = torch.exp(-self.alpha * torch.clamp(cos - self.mrg, min=-10, max=10))
        neg_exp = torch.exp(self.alpha * torch.clamp(cos + self.mrg, min=-10, max=10))

        # Handle the case where there are no valid proxies
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)

        epsilon = 1e-8  # Small epsilon to avoid division by zero
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / (num_valid_proxies + epsilon)
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss