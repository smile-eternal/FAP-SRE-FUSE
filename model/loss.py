import torch.nn as nn
class L1_loss(nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()
        self.criterion=nn.L1Loss()
    def forward(self,x,label):
        return self.criterion(x,label)
class bce_loss(nn.Module):
    def __init__(self):
        super(bce_loss, self).__init__()
        self.criterion=nn.BCELoss()
    def forward(self,x,label):
        return self.criterion(x,label)
