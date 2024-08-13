from model import set_model1
from model import set_module2
from model import attention_module
from model import network_srresnet
import torch
import numpy as np
import torch.nn as nn
import cv2
from skimage import morphology
from torchvision.utils import save_image


class gather_module(nn.Module):
    def __init__(self,opts,net):
        super(gather_module, self).__init__()
        self.model1=set_model1.Siamese_A(opts)
        self.model2=set_module2.Siamese_B(opts)
        self.attention=net
        self.reconstruct=nn.Sequential(
                                       nn.Conv2d(in_channels=3,out_channels=3,kernel_size=1,stride=1),
                                       nn.ReLU()
        )
    def forward(self,x,y):
        ba,_,a,b=x.shape
        ba,_,c,d=y.shape
        gather=torch.cat((x,y),1)
        xl_out,x_out=self.model1(x)
        yl_out,y_out=self.model2(y)
        out=self.attention(gather).detach()
        x_mask = out[:, 0, :, :]
        y_mask = out[:, 1, :, :]
        x_mask=x_mask.unsqueeze(1)
        y_mask=y_mask.unsqueeze(1)
        out=torch.cat((x_mask,y_mask),1)
        out1 = torch.rand((ba , 2, 2 * a, 2 * b))
        for i in range(ba):
            output=out[i,:,:,:]
            output=output.cpu()
            output=output.numpy()
            output=np.transpose(output,(1,2,0))
            output=cv2.resize(output,(2*b,2*a))
            output=np.transpose(output,(2,0,1))
            output=torch.from_numpy(output)
            out1[i,:,:,:]=output
        x_mask=out1[:,0,:,:].unsqueeze(1).to(torch.device("cuda"))
        y_mask=out1[:,1,:,:].unsqueeze(1).to(torch.device("cuda"))
        out=x_mask*xl_out+y_mask*yl_out
        out=self.reconstruct(out)
        return x_out,y_out,out,x_mask,y_mask