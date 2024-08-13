import torch.nn as nn
import torch
from model import same_conv
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
#
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class attention(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(attention, self).__init__()
        self.conv1=nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=3,padding=1,stride=1)
        self.relu=nn.ReLU()
    def forward(self, x):
        feature1=self.relu(self.conv1(x))
        return feature1

class attention_module(nn.Module):
    def __init__(self,opts):
        super(attention_module, self).__init__()
        self.SFE=nn.Conv2d(in_channels=6,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.SFE2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1)
        self.conv=nn.Sequential(same_conv.Conv2d(in_channels=6,out_channels=32,kernel_size=5),
                                nn.LeakyReLU(negative_slope=0.2),
                                same_conv.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
                                nn.LeakyReLU(negative_slope=0.2),
                                same_conv.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
                                nn.LeakyReLU(negative_slope=0.2),
                                )
        self.at_number=opts.at_number
        self.ex_feature=nn.ModuleList()
        self.bn = nn.BatchNorm2d(64)
        for i in range(self.at_number):
            self.ex_feature.append(attention(128,64))
        self.descent1=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.descent2=nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1)
        self.descent3=nn.Conv2d(in_channels=16,out_channels=3,kernel_size=1)
        self.descent4=nn.Conv2d(in_channels=3,out_channels=2,kernel_size=1)
    def forward(self, x):
        SFE=self.SFE2(self.SFE(x))
        lw=F.sigmoid(self.conv(x))
        for i in range(self.at_number):
            original=SFE
            SFE=torch.cat((SFE,lw),1)
            SFE=self.bn(F.sigmoid(self.ex_feature[i](SFE)))
            SFE=original+SFE
        out=F.sigmoid(self.descent4(self.descent3(self.descent2(self.descent1(SFE)))))
        return out


class mulattention(nn.Module):
    def __init__(self):
        super(mulattention, self).__init__()
        self.conv_3=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),)
        self.conv_5=nn.Sequential(same_conv.Conv2d(in_channels=3,out_channels=32,kernel_size=5),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  same_conv.Conv2d(in_channels=32,out_channels=64,kernel_size=5),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  same_conv.Conv2d(in_channels=64,out_channels=32,kernel_size=5),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  same_conv.Conv2d(in_channels=32,out_channels=3,kernel_size=5),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.attention_generate=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1,stride=1),
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,padding=1,stride=1),
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1),
                                              nn.Sigmoid())
    def forward(self,x):
        out_3=self.conv_3(x)
        out_5=self.conv_5(x)
        at=out_3*out_5
        att=self.attention_generate(at)
        out_3=out_3+out_3*att
        out_5=out_5+out_5*att
        return out_3,out_5,att


class spatial_attention(nn.Module):
    def __init__(self):
        super(spatial_attention, self).__init__()
        self.conv_3_3_1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv_3_3_2=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv_5_3_1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv_5_3_2=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,padding=1,stride=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.attention_generate3=nn.Sequential(nn.Conv2d(in_channels=2,out_channels=16,kernel_size=3,padding=1,stride=1),
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,padding=1,stride=1),
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1),
                                              nn.Sigmoid())
        self.attention_generate5=nn.Sequential(nn.Conv2d(in_channels=2,out_channels=16,kernel_size=3,padding=1,stride=1),
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,padding=1,stride=1),
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1),
                                              nn.Sigmoid())
    def forward(self,in_3,in_5):
        out_3_1=self.conv_3_3_1(in_3)
        out_3_2=self.conv_3_3_2(in_3)
        out_5_1=self.conv_5_3_1(in_5)
        out_5_2=self.conv_5_3_2(in_5)
        out_3_1_t=out_3_1.transpose(2,3)
        out_5_1_t=out_5_1.transpose(2,3)
        out_att_3=torch.matmul(out_3_2,out_3_1_t)
        out_att_5=torch.matmul(out_5_2,out_5_1_t)
        out3_1=torch.matmul(out_att_3,out_3_1)
        out3_2=torch.matmul(out_att_3,out_3_2)
        out5_1=torch.matmul(out_att_5,out_5_1)
        out5_2=torch.matmul(out_att_5,out_5_2)
        out3_att=self.attention_generate3(torch.cat((out3_1,out3_2),1))
        out5_att=self.attention_generate5(torch.cat((out5_1,out5_2),1))
        out3=in_3+in_3*out3_att
        out5=in_5+in_5*out5_att
        return out3,out5


class channel_attention(nn.Module):
    def __init__(self):
        super(channel_attention, self).__init__()
        self.avgpool=torch.nn.AdaptiveAvgPool2d(1)
        self.fc_3=nn.Sequential(
            nn.Linear(3,3,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(3,3,bias=False),
            nn.Sigmoid()
        )
        self.fc_5=nn.Sequential(
            nn.Linear(3,3,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(3,3,bias=False),
            nn.Sigmoid()
        )
        self.reconstruct_3=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1,stride=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,stride=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,padding=1,stride=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,padding=1,stride=1),
                                         nn.ReLU())
        self.reconstruct_5=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1,stride=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,stride=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,padding=1,stride=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,padding=1,stride=1),
                                         nn.ReLU())
    def forward(self,in_3,in_5,att):
        b,c,_,_=in_3.size()
        out_3=self.avgpool(in_3).view(b,c)
        out_3=self.fc_3(out_3).view(b,c,1,1)
        out_3= torch.mul(in_3,out_3)
        out_3=torch.mul(out_3,att)
        out_5=self.avgpool(in_5).view(b,c)
        out_5=self.fc_5(out_5).view(b,c,1,1)
        out_5=torch.mul(in_5,out_5)
        out_5=torch.mul(out_5,att)
        out_3=in_3+out_3
        out_5=in_5+out_5
        return out_3,out_5


class attention(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(attention, self).__init__()
        self.conv1=nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=3,padding=1,stride=1)
        self.relu=nn.ReLU()
    def forward(self, x):
        feature1=self.relu(self.conv1(x))
        return feature1

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.mulatt=mulattention()
        self.spatt=spatial_attention()
        self.chatt=channel_attention()
        self.SFE=nn.Conv2d(in_channels=12,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.SFE2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1)
        self.conv=nn.Sequential(same_conv.Conv2d(in_channels=12,out_channels=32,kernel_size=5),
                                nn.LeakyReLU(negative_slope=0.2),
                                same_conv.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
                                nn.LeakyReLU(negative_slope=0.2),
                                same_conv.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
                                nn.LeakyReLU(negative_slope=0.2),
                                )
        self.ex_feature=nn.ModuleList()
        self.bn = nn.BatchNorm2d(64)
        for i in range(10):
            self.ex_feature.append(attention(128,64))
        self.descent1=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.descent2=nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1)
        self.descent3=nn.Conv2d(in_channels=16,out_channels=3,kernel_size=1)
        self.descent4=nn.Conv2d(in_channels=3,out_channels=2,kernel_size=1)
        self.descent5=nn.Conv2d(in_channels=2,out_channels=2,kernel_size=1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
    def forward(self, x):
        y=x[:,:3,:,:]
        z=x[:,3:,:,:]
        out_3_1,out_5_1,att_1=self.mulatt(y)
        out_3_2,out_5_2,att_2=self.mulatt(z)
        out_3_1,out_5_1=self.spatt(out_3_1,out_5_1)
        out_3_2,out_5_2=self.spatt(out_3_2,out_5_2)
        out_3_1,out_5_1=self.chatt(out_3_1,out_5_1,att_1)
        out_3_2,out_5_2=self.chatt(out_3_2,out_5_2,att_2)
        y=torch.cat((out_3_1,out_5_1),1)
        z=torch.cat((out_3_2,out_5_2),1)
        x=torch.cat((y,z),1)
        SFE=self.SFE2(self.SFE(x))
        lw=F.sigmoid(self.conv(x))
        for i in range(10):
            original=SFE
            SFE=torch.cat((SFE,lw),1)
            SFE=self.bn(F.sigmoid(self.ex_feature[i](SFE)))
            SFE=original+SFE
        out=F.sigmoid(self.descent5(self.descent4(self.descent3(self.descent2(self.descent1(SFE))))))
        return out
