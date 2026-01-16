import torch
import torch.nn as nn
import torch.nn.functional as F
# from functorch import jacrev, vmap
from torch.func import jacrev
from torch import vmap
import numpy as np

def mlp(sizes, hid_nonliear, out_nonliear):
    # declare layers
    layers = []
    for j in range(len(sizes) - 1):
        nonliear = hid_nonliear if j < len(sizes) - 2 else out_nonliear
        layers += [nn.Linear(sizes[j], sizes[j + 1]), nonliear()]
    # init weight
    for i in range(len(layers) - 1):
        if isinstance(layers[i], nn.Linear):
            if isinstance(layers[i+1], nn.ReLU):
                nn.init.kaiming_normal_(layers[i].weight, nonlinearity='relu')
            elif isinstance(layers[i+1], nn.LeakyReLU):
                nn.init.kaiming_normal_(layers[i].weight, nonlinearity='leaky_relu')
            else:
                nn.init.xavier_normal_(layers[i].weight)
    return nn.Sequential(*layers)


class K_net(nn.Module):
    def __init__(self, global_lips, k_init, sizes, hid_nonliear, out_nonliear) -> None:
        super().__init__()
        
        self.global_lips = global_lips
        if global_lips == 1:
            print("LipsNet-G")
            # declare global Lipschitz constant
            self.k = torch.nn.Parameter(torch.tensor(k_init, dtype=torch.float), requires_grad=True)
        elif global_lips == 0:
            print("LipsNet-L")
            # declare network
            self.k = mlp(sizes, hid_nonliear, out_nonliear)
            # set K_init
            self.k[-2].bias.data += torch.tensor(k_init, dtype=torch.float).data

    def forward(self, x):
        if self.global_lips == 1:
            return F.softplus(self.k).repeat(x.shape[0]).unsqueeze(1)
        elif self.global_lips == 0:
            return self.k(x)
        

class LipsNet(nn.Module):
    def __init__(self, f_sizes, f_hid_nonliear=nn.ReLU, f_out_nonliear=nn.Identity,
                 global_lips=1, k_init=100, k_sizes=None, k_hid_act=nn.Tanh, k_out_act=nn.Softplus,
                 loss_lambda=0.1, eps=1e-4, squash_action=1) -> None:
        super().__init__()
        
        # declare network
        self.model = mlp(f_sizes, f_hid_nonliear, f_out_nonliear)
        self.k_net = K_net(global_lips, k_init, k_sizes, k_hid_act, k_out_act)
        # declare hyperparameters
        self.loss_lambda = loss_lambda
        self.eps = eps
        self.squash_action = squash_action
        # initialize as eval mode
        self.output_dim = f_sizes[-1]
        self.eval()


    def forward(self, x, state=None, info=None):
        
        # if isinstance(x, np.ndarray):
        #     x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()  # This shares memory with the NumPy array
            x = x.to(next(self.parameters()).device)
    
        # K(x) forward
        k_out = self.k_net(x)
        # L2 regularization backward
        if self.training and k_out.requires_grad:
            lips_loss = self.loss_lambda * (k_out ** 2).mean()
            lips_loss.backward(retain_graph=True)
        # f(x) forward
        f_out = self.model(x)
        # calcute jac matrix
        if k_out.requires_grad:
            jacobi = vmap(jacrev(self.model))(x)
        else:
            with torch.no_grad():
                jacobi = vmap(jacrev(self.model))(x)
        # jacobi.dim: (x.shape[0], f_out.shape[1], x.shape[1])
        #             (batch     , f output dim  , x feature dim)
        # calcute jac norm
        jac_norm = torch.norm(jacobi, 2, dim=(1,2)).unsqueeze(1)
        # multi-dimensional gradient normalization (MGN)
        action = k_out * f_out / (jac_norm + self.eps)
        # squash action
        if self.squash_action == 1:
            action = torch.tanh(action)
        return action, None
