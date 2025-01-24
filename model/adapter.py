import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, D_features,out_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features) #Linear(in_features=768, out_features=192, bias=True)
        self.D_fc2 = nn.Linear(D_hidden_features, out_features) #Linear(in_features=192, out_features=768, bias=True)
        
    def forward(self, x): ## x:[784, 4, 8, 768]
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x) ##[784, 4, 8, 192]
        xs = self.act(xs) 
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x