import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from setting import setting as setn

def get_loss(name):
    if name == "Arcface":
        return Arcface(setn.embedding_size, setn.num_class)
    elif name == "Cosface":
        return Cosface(setn.embedding_size, setn.num_class)
    else:
        raise ValueError()
    

class Arcface(nn.Module):    
    def __init__(self, num_class, num_feature, s = 64.0, m = 0.5):
        super().__init__()
        self.num_class = num_class
        self.num_feature = num_feature        
        self.s = s
        self.m = m
    
        self.header = Parameter(torch.FloatTensor(num_feature, num_class))
        torch.nn.init.normal_(self.header)
        
    def forward(self, feature, label):
        feature = F.normalize(feature)
        logits = F.linear(feature, self.header)
        logits = logits.view(setn.batch_size, -1)
        loss_v = 0
        
        for batch in range(setn.batch_size):
            # add margin
            logits[batch] /= torch.sum(torch.abs(logits[batch].clone()))
            temp = torch.arccos(logits[batch, label[batch]].clone()) + self.m
            logits[batch, label[batch]] = torch.cos(temp)  
            # get loss
            loss = -1 / setn.batch_size * torch.log(torch.exp(self.s * logits[batch, label[batch]]) / torch.sum(torch.exp(self.s * logits[batch])))
            loss_v += loss
            
        return loss_v
        
        
class Cosface(nn.Module):    
    def __init__(self, num_class, num_feature, s = 64.0, m = 0.35):
        super().__init__()
        self.num_class = num_class
        self.num_feature = num_feature  
        self.s = s
        self.m = m
        
        self.header = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.normal_(self.header)
        
    def forward(self, input, label):
        logits = F.linear(feature, self.header)
        logits = F.normalize(logits)
        logits[label] -= self.m
        loss = -1 / setn.batch_size * torch.log(torch.exp(self.s * (logits[label]) - self.m)) / torch.sum(torch.exp(self.s * logits))
        
        return loss
        
        
        
        
        
        
        
        
