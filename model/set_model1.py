import torch.nn as nn
import torch
import numpy as np
import cv2
from torch.autograd import Variable
from torchvision.utils import save_image


def nn_conv2d(im):
    conv_op = nn.Conv2d(1, 1, 3,padding=1, bias=False)
    conv_op.to(device=torch.device("cuda"))
    sobel_kernel = np.array([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 3, 3, 3))
    conv_op.weight.data = torch.from_numpy(sobel_kernel).to(device=torch.device("cuda"))
    edge_detect = conv_op(im)
    edge_detect = edge_detect.detach()
    return edge_detect

class one_conv1(nn.Module):
    def __init__(self,inchannels,outchannels,kernalsize=3):
        super(one_conv1, self).__init__()
        self.conv=nn.Conv2d(in_channels=inchannels,out_channels=outchannels,kernel_size=kernalsize,padding=kernalsize>>1,stride=1)
        self.relu=nn.ReLU()
    def forward(self, x):
        output=self.relu(self.conv(x))
        return torch.cat((x,output),1)
class rdb1(nn.Module):
    def __init__(self,in_channel,out_channel,layer_number):
        super(rdb1, self).__init__()
        layer=[]
        for i in range(layer_number):
            layer.append(one_conv1(inchannels=2*in_channel+out_channel*i,outchannels=out_channel))
        self.conv=nn.Sequential(*layer)
        self.LFF=nn.Conv2d(in_channels=2*in_channel+out_channel*layer_number,out_channels=in_channel,kernel_size=1,padding=0,stride=1)
    def forward(self, x):
        out=self.conv(x)
        lff=self.LFF(out)
        return lff
class rdb2(nn.Module):
    def __init__(self,in_channel,out_channel,layer_number):
        super(rdb2, self).__init__()
        layer=[]
        for i in range(layer_number):
            layer.append(one_conv1(inchannels=in_channel+out_channel*i,outchannels=out_channel))
        self.conv=nn.Sequential(*layer)
        self.LFF=nn.Conv2d(in_channels=in_channel+out_channel*layer_number,out_channels=in_channel,kernel_size=1,padding=0,stride=1)
    def forward(self, x):
        out=self.conv(x)
        lff=self.LFF(out)
        return lff


class Siamese_A(nn.Module):
    def __init__(self,opts):
        super(Siamese_A, self).__init__()
        self.in_channel=opts["MODEL"]["G"]["IN_CHANNELS"]
        self.out_channel=opts["MODEL"]["G"]["OUT_CHANNELS"]
        self.rdb_number=opts["MODEL"]["G"]["RDB_NUMBERS"]
        self.layer_number=opts["MODEL"]["G"]["LAYER_NUMBERS"]
        self.first_channel=opts["MODEL"]["G"]["FIRST_CHANNEL"]
        self.kernal_size=opts["MODEL"]["G"]["KERNEL_SIZE"]
        self.SFE1=nn.Conv2d(in_channels=self.first_channel,out_channels=self.in_channel,kernel_size=self.kernal_size,padding=self.kernal_size>>1,stride=1)
        self.SFE1_detect=nn.Conv2d(in_channels=1,out_channels=self.in_channel,kernel_size=self.kernal_size,padding=self.kernal_size>>1,stride=1)
        self.SFE2=nn.Conv2d(in_channels=2*self.in_channel,out_channels=self.in_channel,kernel_size=self.kernal_size,padding=self.kernal_size>>1,stride=1)
        self.SFE2_detect=nn.Conv2d(in_channels=self.in_channel,out_channels=self.in_channel,kernel_size=self.kernal_size,padding=self.kernal_size>>1,stride=1)
        self.rdn1=nn.ModuleList()
        self.rdn1_detect=nn.ModuleList()
        for i in range(self.rdb_number):
            self.rdn1.append(rdb1(self.in_channel,self.out_channel,self.layer_number))
        for i in range(self.rdb_number):
            self.rdn1_detect.append(rdb2(self.in_channel, self.out_channel, self.layer_number))
        self.GFF=nn.Sequential(nn.Conv2d(self.in_channel*self.rdb_number+2*self.in_channel,self.in_channel,kernel_size=1,padding=0,stride=1),
                               nn.Conv2d(self.in_channel,self.in_channel,kernel_size=self.kernal_size,padding=self.kernal_size>>1,stride=1))
        if opts["MODEL"]["G"]["SCALE"]==2:
            self.up_net=nn.Sequential(nn.Conv2d(self.in_channel,self.in_channel,kernel_size=3,padding=1,stride=1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2d(self.in_channel,self.in_channel,kernel_size=3,padding=1,stride=1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.ConvTranspose2d(in_channels=self.in_channel,out_channels=self.in_channel,kernel_size=3,stride=2,padding=1,output_padding=1),  #×2
                                      # nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.in_channel,kernel_size=3, stride=3, padding=1, output_padding=2),  # ×3
                                      # nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.in_channel,kernel_size=3, stride=2, padding=1, output_padding=1),  # ×4
                                      nn.Conv2d(self.in_channel,self.in_channel,kernel_size=3,padding=1,stride=1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2d(self.in_channel,out_channels=16,kernel_size=3,padding=1,stride=1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,padding=1,stride=1))
        elif opts["MODEL"]["G"]["SCALE"]==3:
            self.up_net=nn.Sequential(nn.Conv2d(self.in_channel,self.in_channel,kernel_size=3,padding=1,stride=1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2d(self.in_channel,self.in_channel,kernel_size=3,padding=1,stride=1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.ConvTranspose2d(in_channels=self.in_channel,out_channels=self.in_channel,kernel_size=3,stride=2,padding=1,output_padding=1),  #×2
                                      nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.in_channel,kernel_size=3, stride=3, padding=1, output_padding=2),  # ×3
                                      # nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.in_channel,kernel_size=3, stride=2, padding=1, output_padding=1),  # ×4
                                      nn.Conv2d(self.in_channel,self.in_channel,kernel_size=3,padding=1,stride=1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2d(self.in_channel,out_channels=16,kernel_size=3,padding=1,stride=1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,padding=1,stride=1))
        elif opts["MODEL"]["G"]["SCALE"]==4:
            self.up_net=nn.Sequential(nn.Conv2d(self.in_channel,self.in_channel,kernel_size=3,padding=1,stride=1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2d(self.in_channel,self.in_channel,kernel_size=3,padding=1,stride=1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.ConvTranspose2d(in_channels=self.in_channel,out_channels=self.in_channel,kernel_size=3,stride=2,padding=1,output_padding=1),  #×2
                                      nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.in_channel,kernel_size=3, stride=3, padding=1, output_padding=2),  # ×3
                                      nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.in_channel,kernel_size=3, stride=2, padding=1, output_padding=1),  # ×4
                                      nn.Conv2d(self.in_channel,self.in_channel,kernel_size=3,padding=1,stride=1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2d(self.in_channel,out_channels=16,kernel_size=3,padding=1,stride=1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,padding=1,stride=1))
        self.deconv=nn.Sequential(nn.Conv2d(in_channels=4,out_channels=3,kernel_size=3,padding=1,stride=1))

        self.attention_generate=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1,stride=1),
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,padding=1,stride=1),
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=16,out_channels=1,kernel_size=1),
                                              nn.Sigmoid())
        self.reconstruct=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,padding=1,stride=1)
                                       )

    def forward(self, x):
        a,_,b,c=x.shape
        out_hgradient = torch.rand((a, 1, 2 * b, 2 * c)).to(torch.device("cuda"))
        out_lgradient = torch.rand((a, 1, b, c)).to(torch.device("cuda"))
        for i in range(a):
            output=x[i,:,:,:].unsqueeze(0)
            gradient = nn_conv2d(output).squeeze(0)
            output=gradient.cpu()
            output=output.numpy()
            output=np.transpose(output,(1,2,0))
            output=cv2.resize(output,(2*c,2*b))
            output=torch.from_numpy(output).unsqueeze(0)
            out_hgradient[i,:,:,:]=output
            out_lgradient[i,:,:,:]=gradient
        save_image(out_hgradient,'gra.png')

        out=self.SFE1(x)
        out_lgradient=self.SFE1_detect(out_lgradient)
        out=torch.cat((out,out_lgradient),1)
        out=self.SFE2(out)
        out_lgradient=self.SFE2_detect(out)
        out1=torch.cat((out,out_lgradient),1)
        shallow=out
        rdb_out=[]
        for i in range(self.rdb_number):
            out_lgradient=self.rdn1_detect[i](out_lgradient)
            out1=self.rdn1[i](out1)
            rdb_out.append(out1)
            deep = out1
            out1=torch.cat((out_lgradient,out1),1)

        lff=torch.cat(rdb_out,1)
        lff=torch.cat((lff,out1),1)
        lff=self.GFF(lff)

        lff=lff+out
        att=self.attention_generate(torch.cat((shallow,deep),1))
        lff=lff*att+lff
        # output1=lff
        output1=self.deconv(torch.cat((self.up_net(lff),out_hgradient),1))
        output2=self.reconstruct(output1)
        return output1,output2







