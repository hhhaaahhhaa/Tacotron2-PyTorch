import os
import numpy as np
import json
import torch
import torch.nn as nn


class ResemblyzerTable(nn.Module):
    def __init__(self, preprocessed_data_dir, dim):
        super().__init__()
        with open(f"{preprocessed_data_dir}/speakers.json", "r", encoding="utf-8") as f:
            self.speakers = json.load(f)
        dvectors = np.zeros((len(self.speakers), dim))
        for spk, idx in self.speakers.items():
            dvectors[idx] = np.load(f"{preprocessed_data_dir}/dvector/{spk}.npy")
        self.model = nn.Embedding.from_pretrained(torch.from_numpy(dvectors).float(), padding_idx=0)
    
    def speaker_name2id(self, name):
        return self.speakers[name]
    
    def forward(self, x):
        return self.model(x)


class SpeakerTable(nn.Module):
    def __init__(self, preprocessed_data_dir, dim):
        super().__init__()
        with open(f"{preprocessed_data_dir}/speakers.json", "r", encoding="utf-8") as f:
            self.speakers = json.load(f)
        self.model = nn.Embedding(len(self.speakers), dim)
        
    def speaker_name2id(self, name):
        return self.speakers[name]
    
    def forward(self, x):
        return self.model(x)
