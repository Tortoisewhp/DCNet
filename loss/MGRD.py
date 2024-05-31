import torch
import torch.nn as nn
from torch.nn import functional as F
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class GCN(nn.Module):
    def __init__(self, plane):
        super(GCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, x):
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return nn.Sigmoid()(out)


def spatialRelation(featureX,featureY):
    b,c,h,w=featureX.shape
    feature1=featureX.reshape(b,c,-1)
    feature2=featureY.reshape(b,h*w,-1)
    out=torch.bmm(feature2,feature1)
    out=torch.softmax(out,2)
    return out

class mgr(nn.Module):
    def __init__(self):
        super(mgr, self).__init__()
        b, c, h, w = yFusedFeature.shape
        self.Transy1 = nn.Conv2d(c, 1, 1, 1, 0)
        self.Transy2 = nn.Conv2d(3, 3, 1, 1, 0)
        self.gcmy1 = GCM(1, 1)
        self.gcmy3 = GCM(1, 1)
        self.gcmy5 = GCM(1, 1)
        self.gcny = GCN(3)
    def forward(self,x):
        x=self.Transy1(x)
        x1=self.gcmy1(x)
        x2=self.gcmy3(x)
        x3=self.gcmy5(x)
        x=self.Transy2(torch.cat([x1,x2,x3],dim=1))
        x=self.gcny(x)
        r=spatialRelation(x,x)
        return r

def MGRD(yFusedFeature,YFusedFeature):
    mgry=mgr()
    mgrY=mgr()
    loss=nn.MSELoss()(mgry(yFusedFeature),mgrY(YFusedFeature))
    return loss

if __name__ == '__main__':
    yFusedFeature=torch.randn(2,512,30,40)
    YFusedFeature = torch.randn(2,512,30,40)
    print(MGRD(yFusedFeature,YFusedFeature))