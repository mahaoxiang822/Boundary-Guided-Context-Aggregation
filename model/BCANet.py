'''
Author: Haoxiang Ma
'''

import torch
import torch.nn.functional as F
from torch import nn
from model.dilated_resnet import get_resnet50_baseline, get_resnet101_baseline


class BCA(nn.Module):
    def __init__(self, xin_channels, yin_channels, mid_channels, BatchNorm=nn.BatchNorm2d, scale=False):
        super(BCA, self).__init__()
        self.mid_channels = mid_channels
        self.f_self = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels=yin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=xin_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(xin_channels),
        )
        self.scale = scale
        nn.init.constant_(self.f_up[1].weight, 0)
        nn.init.constant_(self.f_up[1].bias, 0)

    def forward(self, x, y):
        batch_size = x.size(0)
        fself = self.f_self(x).view(batch_size, self.mid_channels, -1)
        fself = fself.permute(0, 2, 1)
        fx = self.f_x(x).view(batch_size, self.mid_channels, -1)
        fx = fx.permute(0, 2, 1)
        fy = self.f_y(y).view(batch_size, self.mid_channels, -1)
        sim_map = torch.matmul(fx, fy)
        if self.scale:
            sim_map = (self.mid_channels ** -.5) * sim_map
        sim_map_div_C = F.softmax(sim_map, dim=-1)
        fout = torch.matmul(sim_map_div_C, fself)
        fout = fout.permute(0, 2, 1).contiguous()
        fout = fout.view(batch_size, self.mid_channels, *x.size()[2:])
        out = self.f_up(fout)
        return x + out



class BCANet(nn.Module):

    def __init__(self, num_classes, BatchNorm=nn.BatchNorm2d, layers=50, multi_grid=(1, 1, 1), criterion=None,
                 pretrained=True):

        super(BCANet, self).__init__()
        self.criterion = criterion
        self.num_classes = num_classes
        self.BatchNorm = BatchNorm
        if layers == 50:
            resnet = get_resnet50_baseline(pretrained=pretrained, num_classes=num_classes, BatchNorm=BatchNorm,
                                           multi_grid=multi_grid)
        elif layers == 101:
            resnet = get_resnet101_baseline(pretrained=pretrained, num_classes=num_classes, BatchNorm=BatchNorm,
                                            multi_grid=multi_grid)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.interpolate = F.interpolate
        del resnet
        self.att = BCA(2048, 256, 256, BatchNorm)

        self.edge_conv1 = self.generate_edge_conv(256)
        self.edge_out1 = nn.Sequential(nn.Conv2d(256,1,1),
                                       nn.Sigmoid())
        self.edge_conv2 = self.generate_edge_conv(512)
        self.edge_out2 = nn.Sequential(nn.Conv2d(256, 1, 1),
                                       nn.Sigmoid())
        self.edge_conv3 = self.generate_edge_conv(1024)
        self.edge_out3 = nn.Sequential(nn.Conv2d(256, 1, 1),
                                       nn.Sigmoid())
        self.edge_conv4 = self.generate_edge_conv(2048)
        self.edge_out4 = nn.Sequential(nn.Conv2d(256, 1, 1),
                                       nn.Sigmoid())

        self.down = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
        )
        self.final_seg = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))
        self.aux_seg = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

    def generate_edge_conv(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, 3),
            self.BatchNorm(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, inp, gts=None):

        x_size = inp.size()

        # res 1
        m0 = self.layer0(inp)
        m1 = self.layer1(m0)
        m2 = self.layer2(m1)
        m3 = self.layer3(m2)
        m4 = self.layer4(m3)

        # MSB
        e1 = self.edge_conv1(m1)
        e1 = self.interpolate(e1, m4.size()[2:], mode='bilinear', align_corners=True)
        e1_out = self.edge_out1(e1)
        e2 = self.edge_conv2(m2)
        e2 = self.interpolate(e2, m4.size()[2:], mode='bilinear', align_corners=True)
        e2_out = self.edge_out2(e2)
        e3 = self.edge_conv3(m3)
        e3 = self.interpolate(e3, m4.size()[2:], mode='bilinear', align_corners=True)
        e3_out = self.edge_out3(e3)
        e4 = self.edge_conv4(m4)
        e4 = self.interpolate(e4, m4.size()[2:], mode='bilinear', align_corners=True)
        e4_out = self.edge_out4(e4)
        e = torch.cat((e1, e2, e3, e4), dim=1)
        e = self.down(e)
        bound_out_ = e1_out+e2_out+e3_out+e4_out

        # BCA
        out_feature = self.att(m4, e)

        seg_out_ = self.final_seg(out_feature)
        seg_out = self.interpolate(seg_out_, x_size[2:], mode='bilinear', align_corners=True)
        bound_out = self.interpolate(bound_out_, x_size[2:], mode='bilinear', align_corners=True)
        if self.training:
            aux_seg_out = self.interpolate(self.aux_seg(m3), x_size[2:], mode='bilinear', align_corners=True)
            return seg_out, bound_out, self.criterion((seg_out,aux_seg_out, bound_out), gts)
        else:
            return seg_out, bound_out





