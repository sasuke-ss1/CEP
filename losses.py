import torch
import torch.nn as nn

class GrayLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        meanRgb = torch.mean(x, [2, 3], keepdim=True)
        ur, ug, ub = torch.split(meanRgb, 1, dim=1)
        
        Dr = torch.pow(ur-0.5, 2)
        Dg = torch.pow(ug-0.5, 2)
        Db = torch.pow(ub-0.5, 2)
        
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Dg, 2) + torch.pow(Db, 2), 0.5)
        
        return k.mean()