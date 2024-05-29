import datetime
import os
import random
from pprint import pprint

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn 
import torch.nn.init as init

def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
                
def dgmg_message_weight_init(m):
    
    def _weight_init(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight.data, std=1.0/10)
            init.normal_(m.bias.data, std=1.0/10)
        else:
            raise ValueError('Expected the input to be of type nn.Linear!')
        
    if isinstance(m, nn.ModuleList):
        for layer in m:
            layer.apply(_weight_init)
    
    else:
        m.apply(_weight_init)