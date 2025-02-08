import torch.nn as nn
import torch

class QuantileLoss(nn.Module):
    def __init__(self,alpha:float=0.15):
        super().__init__()
        self.alpha=alpha
    
    def forward(self,y,y_hat,N=None):
        if not N:
            Loss=(torch.sum(self.alpha*(y-y_hat)*(y>y_hat))+torch.sum((self.alpha-1)*(y-y_hat)*(y<=y_hat)))/2
        else:
            assert y.shape==N.shape,'the shape of N is not compatible with y'
            Loss=(torch.sum((self.alpha-N)*(y-y_hat)*(y>y_hat))+torch.sum((self.alpha+N-1)*(y-y_hat)*(y<=y_hat)))/2

        return Loss