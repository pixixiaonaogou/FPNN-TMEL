from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class conv_sknet(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=16,kernel_three=False):
        super(conv_sknet, self).__init__()
        self.kernel_three = kernel_three
        
        def register(name, tensor):
            self.register_parameter(name, Parameter(tensor))

        register('A', torch.rand(max(out_ch // reduction, 32), out_ch))
        register('B', torch.rand(max(out_ch // reduction, 32), out_ch))
        
#        register('SA',torch.rand(max(out_ch//reduction, 32), out_ch, size, size))
#        register('SB',torch.rand(max(out_ch//reduction, 32), out_ch, size, size))
        
        if self.kernel_three:
            register('C', torch.rand(max(out_ch // reduction, 32), out_ch))
            self.conv_7 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, dilation=3, padding=3),
            SynchronizedBatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            )    

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            SynchronizedBatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, dilation=2, padding=2),
            SynchronizedBatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        

        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
                nn.Linear(out_ch, max(out_ch // reduction, 32)),
                nn.ReLU(inplace=True),
        )

#        self.spatial_se = nn.Sequential(nn.Conv2d(out_ch, 1, kernel_size=1,
#                                          stride=1, padding=0, bias=False),
#                                nn.Sigmoid())
    def forward(self, x):
#        x_1 = self.conv_1(x)
        if not self.kernel_three:
            x_3 = self.conv_3(x)
            x_5 = self.conv_5(x)
            x_fuse = x_3 + x_5

            b, c, _, _ = x_fuse.size()  # b=4, c=128
            x_fuse_s = self.avg_pool(x_fuse).view(b, c)
            x_fuse_z = self.fc(x_fuse_s)

            s1 = torch.Tensor(np.exp(np.array(torch.mm(x_fuse_z, self.A).cpu().detach().numpy())) / (np.exp(np.array(torch.mm(x_fuse_z, self.A).cpu().detach().numpy())) + np.exp(np.array(torch.mm(x_fuse_z, self.B).cpu().detach().numpy()))))
            s1 = s1.view(b, c, 1, 1).cuda()
            s2 = 1 - s1
            V_a = x_3 * s1
            V_b = x_5 * s2
            V = V_a + V_b
        
        else:
            x_3 = self.conv_3(x)
            x_5 = self.conv_5(x)
            x_7 = self.conv_7(x)
            x_fuse = x_3 + x_5 + x_7
            b, c, _, _ = x_fuse.size()  # b=4, c=128
            x_fuse_s = self.avg_pool(x_fuse).view(b, c)
            x_fuse_z = self.fc(x_fuse_s)
    
            s1 = torch.Tensor(np.exp(np.array(torch.mm(x_fuse_z, self.A).cpu().detach().numpy())) / (np.exp(np.array(torch.mm(x_fuse_z, self.C).cpu().detach().numpy())) + np.exp(np.array(torch.mm(x_fuse_z, self.A).cpu().detach().numpy())) + np.exp(np.array(torch.mm(x_fuse_z, self.B).cpu().detach().numpy()))))
            s2 = torch.Tensor(np.exp(np.array(torch.mm(x_fuse_z, self.B).cpu().detach().numpy())) / (np.exp(np.array(torch.mm(x_fuse_z, self.C).cpu().detach().numpy())) + np.exp(np.array(torch.mm(x_fuse_z, self.A).cpu().detach().numpy())) + np.exp(np.array(torch.mm(x_fuse_z, self.B).cpu().detach().numpy()))))
            s1 = s1.view(b, c, 1, 1).cuda()
            s2 = s2.view(b, c, 1, 1).cuda()
            s3 = 1 - s1 - s2
            V_a = x_3 * s1
            V_b = x_5 * s2
            V_c = x_7 * s3
            V = V_a + V_b + V_c        
        
        
        
        return V

class dual_attention_conv_sknet(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=16,kernel_three=False,size=8,batch_size=8):
        super(dual_attention_conv_sknet, self).__init__()
        self.kernel_three = kernel_three
        self.batch_size = batch_size 
        self.size = size
        
        def register(name, tensor):
            self.register_parameter(name, Parameter(tensor))

        register('A', torch.rand(max(out_ch // reduction, 32), out_ch))
        register('B', torch.rand(max(out_ch // reduction, 32), out_ch))
        
       # register('SA',torch.rand( self.batch_size, 1, size, size))
       # register('SB',torch.rand( self.batch_size, 1, size, size))
                
        if self.kernel_three:
            register('C', torch.rand(max(out_ch // reduction, 32), out_ch))
            self.conv_7 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, dilation=3, padding=3),
            SynchronizedBatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            )    

        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            SynchronizedBatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, dilation=2, padding=2),
            SynchronizedBatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
       
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
                nn.Linear(out_ch, max(out_ch // reduction, 32)),
                nn.ReLU(inplace=True),
        )
        
        self.spatial_se = nn.Sequential(nn.Conv2d(out_ch, 1, kernel_size=1,
                                          stride=1, padding=0, bias=False),
                                nn.Sigmoid())        
        

        

    def forward(self, x):
#        x_1 = self.conv_1(x)
        if not self.kernel_three:
            x_3 = self.conv_3(x)
            x_5 = self.conv_5(x)
            x_fuse = x_3 + x_5
            
            
            
            
            # channel attention
            b, c, _, _ = x_fuse.size()  # b=4, c=128
            x_fuse_s = self.avg_pool(x_fuse).view(b, c)
            x_fuse_z = self.fc(x_fuse_s)
    
            s1 = torch.Tensor(np.exp(np.array(torch.mm(x_fuse_z, self.A).cpu().detach().numpy())) / (np.exp(np.array(torch.mm(x_fuse_z, self.A).cpu().detach().numpy())) + np.exp(np.array(torch.mm(x_fuse_z, self.B).cpu().detach().numpy()))))
            s1 = s1.view(b, c, 1, 1).cuda()
            s2 = 1 - s1
            V_a = x_3 * s1
            V_b = x_5 * s2
            V = V_a + V_b
   
    
            #spatial attention
            x_fuse_sa = self.spatial_se(x_fuse)   
            
            SA = torch.rand(b,1,self.size,self.size).cuda()
            SB = torch.rand(b,1,self.size,self.size).cuda()
            #print(type(SA))
           # print(type(SB))
            #sa1 = torch.Tensor(np.exp(np.array(torch.matmul(x_fuse_sa, self.SA).cpu().detach().numpy())) / (np.exp(np.array(torch.matmul(x_fuse_sa, self.SA).cpu().detach().numpy())) + np.exp(np.array(torch.matmul(x_fuse_sa, self.SB).cpu().detach().numpy()))))
            sa1 = torch.Tensor((np.array(torch.matmul(x_fuse_sa, SA).cpu().detach().numpy())) / ((np.array(torch.matmul(x_fuse_sa, SA).cpu().detach().numpy())) + (np.array(torch.matmul(x_fuse_sa, SB).cpu().detach().numpy()))))

            sa1 = sa1.cuda()
            sa2 = 1 - sa1
            SV_a = x_3 * sa1
            SV_b = x_5 * sa2
            SV = SV_a + SV_b

            return V  + SV
     
        else:
            x_3 = self.conv_3(x)
            x_5 = self.conv_5(x)
            x_7 = self.conv_7(x)
            x_fuse = x_3 + x_5 + x_7
            b, c, _, _ = x_fuse.size()  # b=4, c=128
            x_fuse_s = self.avg_pool(x_fuse).view(b, c)
            x_fuse_z = self.fc(x_fuse_s)
    
            s1 = torch.Tensor(np.exp(np.array(torch.mm(x_fuse_z, self.A).cpu().detach().numpy())) / (np.exp(np.array(torch.mm(x_fuse_z, self.C).cpu().detach().numpy())) + np.exp(np.array(torch.mm(x_fuse_z, self.A).cpu().detach().numpy())) + np.exp(np.array(torch.mm(x_fuse_z, self.B).cpu().detach().numpy()))))
            s2 = torch.Tensor(np.exp(np.array(torch.mm(x_fuse_z, self.B).cpu().detach().numpy())) / (np.exp(np.array(torch.mm(x_fuse_z, self.C).cpu().detach().numpy())) + np.exp(np.array(torch.mm(x_fuse_z, self.A).cpu().detach().numpy())) + np.exp(np.array(torch.mm(x_fuse_z, self.B).cpu().detach().numpy()))))
            s1 = s1.view(b, c, 1, 1).cuda()
            s2 = s2.view(b, c, 1, 1).cuda()
            s3 = 1 - s1 - s2
            V_a = x_3 * s1
            V_b = x_5 * s2
            V_c = x_7 * s3
            V = V_a + V_b + V_c        
        
        
        
            return V



class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inconv, self).__init__()
        self.conv = nn.Sequential(
            conv_sknet(in_ch, out_ch),
            conv_sknet(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_sknet(in_ch, out_ch),
            conv_sknet(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up1, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = conv_sknet(2 * in_ch, in_ch)
        self.conv2 = conv_sknet(in_ch, out_ch)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  #  order matters?
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Up2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up2, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = conv_sknet(1024, 2 * out_ch)
        self.conv2 = conv_sknet(2 * out_ch, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Up3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up3, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = conv_sknet(512, 2 * out_ch)
        self.conv2 = conv_sknet(2 * out_ch, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x2 = nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x3, x2], dim=1)  # order matters?
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Up4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up4, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = conv_sknet(256, out_ch)
        self.conv2 = conv_sknet(out_ch, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)  # 1*1 conv kernal

    def forward(self, x):
        x = self.conv(x)
        return x