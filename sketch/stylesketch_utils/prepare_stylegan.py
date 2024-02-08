import sys
sys.path.append('..')
from collections import OrderedDict
import torch
import torch.nn as nn
import dnnlib
from models.networks_stylegan2 import MappingNetwork,SynthesisNetwork
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_stylegan(_, saved_dir):
    avg_latent_dir = saved_dir.rsplit('/',1)[0]+"/avg.pt"
    avg_latent = torch.load(avg_latent_dir)
    
    g_all = nn.Sequential(OrderedDict([
        ('g_mapping', MappingNetwork(512,0,512,18)),
        ('g_synthesis', SynthesisNetwork(512,1024,3))
    ]))
    
    g_all.load_state_dict(torch.load(saved_dir, map_location=device))
    g_all.eval().cuda()


    return g_all, avg_latent