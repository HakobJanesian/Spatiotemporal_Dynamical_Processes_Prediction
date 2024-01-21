import torch.nn as nn
import torch
from collections import OrderedDict
import torch.nn.init as init
import numpy as np

class FCN(nn.Module):
    def __init__(self, layers_list, activation_function_list = None):
        nn.Module.__init__(self)

        self._depth = len(layers_list) - 1
        if activation_function_list is None:
            activation_function_list = [torch.nn.Softplus for _ in range(self._depth - 1)]        
        
        seq = []
        for i, activation_function in enumerate(activation_function_list):
            seq.append(('layer_%d' % i, torch.nn.Linear(layers_list[i], layers_list[i+1], dtype = torch.float64)))
            seq.append(('activation_%d' % i, activation_function()))
        
        seq.append(('layer_%d' % (self._depth - 1), torch.nn.Linear(layers_list[-2], layers_list[-1], dtype = torch.float64)))
        
        self._Wtmx = torch.nn.Sequential(OrderedDict(seq))
        self.optimizer = torch.optim.Adam( params = self._Wtmx.parameters(), lr=0.01 )    
        
    def forward(self, x):
        return self._Wtmx(x)

class XFCN(nn.Module):
    def __init__(self, layers_list, activation_function_list=None):
        nn.Module.__init__(self)

        self._depth = len(layers_list) - 1
        if activation_function_list is None:
            activation_function_list = [torch.nn.Softplus for _ in range(self._depth - 1)]

        seq = []
        for i, activation_function in enumerate(activation_function_list):
            layer = torch.nn.Linear(layers_list[i], layers_list[i+1], dtype=torch.float64)
            init.xavier_uniform_(layer.weight)  # Apply Xavier initialization to weights
            seq.append(('layer_%d' % i, layer))
            seq.append(('activation_%d' % i, activation_function()))

        layer_last = torch.nn.Linear(layers_list[-2], layers_list[-1], dtype=torch.float64)
        init.xavier_uniform_(layer_last.weight)  # Apply Xavier initialization to weights
        seq.append(('layer_%d' % (self._depth - 1), layer_last))

        self._Wtmx = torch.nn.Sequential(OrderedDict(seq))
        self.optimizer = torch.optim.Adam(params=self._Wtmx.parameters(), lr=0.01)

    def forward(self, x):
        return self._Wtmx(x)

class LP_XFCN(nn.Module):
    def __init__(self, layers_list, activation_function_list=None):
        nn.Module.__init__(self)

        self._depth = len(layers_list) - 1
        if activation_function_list is None:
            activation_function_list = [torch.nn.Softplus for _ in range(self._depth - 1)]

        seq = []
        for i, activation_function in enumerate(activation_function_list):
            layer = torch.nn.Linear(layers_list[i], layers_list[i+1], dtype=torch.float32)
            init.xavier_uniform_(layer.weight)  # Apply Xavier initialization to weights
            seq.append(('layer_%d' % i, layer))
            seq.append(('activation_%d' % i, activation_function()))

        layer_last = torch.nn.Linear(layers_list[-2], layers_list[-1], dtype=torch.float32)
        init.xavier_uniform_(layer_last.weight)  # Apply Xavier initialization to weights
        seq.append(('layer_%d' % (self._depth - 1), layer_last))

        self._Wtmx = torch.nn.Sequential(OrderedDict(seq))
        self.optimizer = torch.optim.Adam(params=self._Wtmx.parameters(), lr=0.01)

    def forward(self, x):
        return self._Wtmx(x)
    
class LLP_XFCN(nn.Module):
    def __init__(self, layers_list, activation_function_list=None):
        nn.Module.__init__(self)

        self._depth = len(layers_list) - 1
        if activation_function_list is None:
            activation_function_list = [torch.nn.Softplus for _ in range(self._depth - 1)]

        seq = []
        for i, activation_function in enumerate(activation_function_list):
            layer = torch.nn.Linear(layers_list[i], layers_list[i+1], dtype=torch.float16)
            init.xavier_uniform_(layer.weight)  # Apply Xavier initialization to weights
            seq.append(('layer_%d' % i, layer))
            seq.append(('activation_%d' % i, activation_function()))

        layer_last = torch.nn.Linear(layers_list[-2], layers_list[-1], dtype=torch.float16)
        init.xavier_uniform_(layer_last.weight)  # Apply Xavier initialization to weights
        seq.append(('layer_%d' % (self._depth - 1), layer_last))

        self._Wtmx = torch.nn.Sequential(OrderedDict(seq))
        self.optimizer = torch.optim.Adam(params=self._Wtmx.parameters(), lr=0.01)

    def forward(self, x):
        return self._Wtmx(x)
    
class MShuffle(nn.Module):
    def __init__(self, exp_size, n_depth = None, again = 1, activation_function_list = None):
        nn.Module.__init__(self)
        if n_depth is None:
            n_depth = exp_size
        
        if activation_function_list is None:
            activation_function_list = [torch.nn.Softplus]*n_depth
        
        seq = []

        for ag in range(again):
            for n, activation_function in enumerate(activation_function_list):
                seq.append(('layer_ag_%d_n_%d' % (ag + 1, n + 1), torch.nn.Linear(2**exp_size, 2**(exp_size-1), dtype = torch.float64)))
                seq.append(('activation_ag_%d_n_%d' % (ag + 1,n + 1), activation_function()))

        self._Wtmx = torch.nn.Sequential(OrderedDict(seq))
        self.n_depth = n_depth

        mask = np.zeros((n_depth,*[2] * exp_size),dtype = bool)
        for d in range(n_depth):
            exec(f"mask[{d},{':,'*d+'0'}] = True")
        self.mask = torch.tensor(mask.reshape(n_depth,-1),dtype=bool) 
        self.again = again

    def forward(self,x):
        Seq = self._Wtmx.children()
        for _ in range(self.again):
            for m in self.mask:
                LM = next(Seq)
                func = next(Seq) 

                x_new = x.clone()
                x_new[:,m] += func(LM(x))
                x = x_new
        return x

class MShufflev2(MShuffle):
    
    def forward(self,x):
        Seq = self._Wtmx.children()
        for _ in range(self.again):
            for m in self.mask:
                LM = next(Seq)
                func = next(Seq) 

                x_new = x.clone()
                x_new[:,m] = func(LM(x))
                x = x_new
        return x
      