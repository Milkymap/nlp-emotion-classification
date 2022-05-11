import torch as th 
import torch.nn as nn 

import functools as ft 

class MLP_Model(nn.Module):
    def __init__(self, layer_cfg, non_linears, dropouts):
        super(MLP_Model, self).__init__()
        self.shapes = zip(layer_cfg[:-1], layer_cfg[1:])
        self.non_linears = non_linears 
        self.dropouts = dropouts 

        self.layers = nn.ModuleList([])

        for (i_dim, o_dim), apply_fn, val in zip(self.shapes, self.non_linears, self.dropouts):
            theta = nn.ReLU() if apply_fn == 1 else nn.Identity()
            proba = nn.Dropout(p=val)
            linear = nn.Linear(i_dim, o_dim)
            block = nn.Sequential(linear, proba, theta)
            self.layers.append(block)
    
    def forward(self, X0):
        XN = ft.reduce(lambda Xi, Li: Li(Xi), self.layers, X0)
        return XN 

if __name__ == '__main__':
    model = MLP_Model(layer_cfg=[384, 128, 64, 1],  non_linears=[1, 1, 0], dropouts=[0.3, 0.2, 0.0])
    print(model)
    X = th.randn((10, 384))
    O = model(X)
    print(O)
    

         

