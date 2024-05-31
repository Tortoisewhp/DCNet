import torch
import torch.nn as nn
from toolbox.models.teacherb4_studentb0_model17_2Net.segformer.mix_transformer import mit_b4
import math
from torch.nn import functional as F

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class DN(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(DN, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0
        self.num_freq = len(mapper_x)
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)
        return dct_filter

class DEA(nn.Module):
    def __init__(self,channel,h,w):
        super(DEA, self).__init__()
        self.channel=channel
        self.Rmlp=nn.Conv2d(self.channel,1,1,1,0)
        self.Dmlp = nn.Conv2d(self.channel, 1, 1, 1, 0)
        self.Amlp = nn.Conv2d(self.channel, 1, 1, 1, 0)
        self.change=nn.Conv2d(3,self.channel, 1, 1, 0)
        self.Rpa=nn.Linear(26*26,self.channel)
        self.Dpa = nn.Linear(26*26,self.channel)
        self.dct=DN(channel,h,w)
        self.dct1 = DN(channel, h, w)

    def forward(self, rgbFeature,depthFeature,Aux):
        b, c, h, w = rgbFeature.shape
        depthFeature=self.dct(depthFeature)
        Aux = F.interpolate(Aux, size=(h, w), mode='bilinear')
        Aux=self.dct1(self.change(Aux))
        thea=self.Rmlp(rgbFeature)
        beta = self.Dmlp(depthFeature)
        Aux=self.Amlp(Aux)
        thea=F.interpolate(thea,size=(26, 26), mode='bilinear')
        beta = F.interpolate(beta, size=(26, 26), mode='bilinear')
        Auxbeta = F.interpolate(Aux, size=(26, 26), mode='bilinear')
        beta=beta*Auxbeta
        rgbthea=nn.Tanh()(thea+beta)
        depthbeta=nn.Tanh()(thea+beta)
        rgbM=torch.matmul(thea.reshape(b,1,26*26).permute(0,2,1),rgbthea.reshape(b,1,26*26))
        depthM = torch.matmul(beta.reshape(b, 1, 26 * 26).permute(0, 2, 1), depthbeta.reshape(b, 1, 26 * 26))
        rgbM=self.Rpa(rgbM).reshape(b,26*26,self.channel).permute(0,2,1).reshape(b,self.channel,26,26)
        depthM=self.Dpa(depthM).reshape(b,26*26,self.channel).permute(0,2,1).reshape(b,self.channel,26,26)
        rgbM = F.interpolate(rgbM, size=(h, w), mode='bilinear')
        depthM = F.interpolate(depthM, size=(h, w), mode='bilinear')
        rgbM=rgbM*rgbFeature+rgbFeature
        depthM=depthM*depthFeature+depthFeature
        fused=nn.Sigmoid()(rgbM*depthM)*rgbFeature+rgbFeature
        return fused

class ECA(nn.Module):
    def __init__(self, in_channel):
        super(ECA, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, padding=0),
        )
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1)),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0))
        )
        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 5), padding=(0, 2)),
            nn.MaxPool2d(kernel_size=(5, 1), padding=(2, 0))
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 7), padding=(0, 3)),
            nn.MaxPool2d(kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 9), padding=(0, 4)),
            nn.MaxPool2d(kernel_size=(9, 1), padding=(4, 0))
        )
        self.branch5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 11), padding=(0, 5)),
            nn.MaxPool2d(kernel_size=(11, 1), padding=(5, 0))
        )
        self.branch6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 13), padding=(0, 6)),
            nn.MaxPool2d(kernel_size=(13, 1), padding=(6, 0))
        )
        self.branch7 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 15), padding=(0, 7)),
            nn.MaxPool2d(kernel_size=(15, 1), padding=(7, 0))
        )
        self.branch8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 17), padding=(0, 8))
        )
        self.branch9 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 19), padding=(0, 9))
        )
        self.branch01 = nn.Sequential(
            nn.AvgPool2d(kernel_size=1, padding=0),
        )
        self.branch11 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 3), padding=(0, 1)),
            nn.AvgPool2d(kernel_size=(3, 1), padding=(1, 0))
        )
        self.branch21 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 5), padding=(0, 2)),
            nn.AvgPool2d(kernel_size=(5, 1), padding=(2, 0))
        )
        self.branch31 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 7), padding=(0, 3)),
            nn.AvgPool2d(kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch41 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 9), padding=(0, 4)),
            nn.AvgPool2d(kernel_size=(9, 1), padding=(4, 0))
        )
        self.branch51 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 11), padding=(0, 5)),
            nn.AvgPool2d(kernel_size=(11, 1), padding=(5, 0))
        )
        self.branch61 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 13), padding=(0, 6)),
            nn.AvgPool2d(kernel_size=(13, 1), padding=(6, 0))
        )
        self.branch71 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 15), padding=(0, 7)),
            nn.AvgPool2d(kernel_size=(15, 1), padding=(7, 0))
        )
        self.branch81 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 17), padding=(0, 8))
        )
        self.branch91 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 19), padding=(0, 9))
        )
        self.conv_cat = nn.Conv2d(20 * in_channel, in_channel, 1, 1, 0)

    def forward(self, x):
        x0 = F.interpolate(self.branch0(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x1 = F.interpolate(self.branch1(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.branch2(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.branch3(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.branch4(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x5 = F.interpolate(self.branch5(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x6 = F.interpolate(self.branch6(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x7 = F.interpolate(self.branch7(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x8 = F.interpolate(self.branch8(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x9 = F.interpolate(self.branch9(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x01 = F.interpolate(self.branch01(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x11 = F.interpolate(self.branch11(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x21 = F.interpolate(self.branch21(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x31 = F.interpolate(self.branch31(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x41 = F.interpolate(self.branch41(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x51 = F.interpolate(self.branch51(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x61 = F.interpolate(self.branch61(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x71 = F.interpolate(self.branch71(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x81 = F.interpolate(self.branch81(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x91 = F.interpolate(self.branch91(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x = self.conv_cat(
            torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x01, x11, x21, x31, x41, x51, x61, x71, x81, x91), 1))
        return x

class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.num_cascade=4
        self.skip_mum=48
        inchannels = [512, 320, 128, 64]
        self.ReduceDemesion=nn.ModuleList([nn.Sequential(
            nn.Conv2d(inchannels[_], inchannels[1+_], 1, 1,0, bias=False)) for _ in range(self.num_cascade-1)])
        self.lowGuides=nn.ModuleList([nn.Sequential(
            nn.Conv2d(channel, inchannels[_], 1, 1,0, bias=False)) for _ in range(self.num_cascade)])
        self.NEMs=nn.ModuleList([nn.MultiheadAttention(inchannels[i],8,0.1) for i in range(self.num_cascade)])
        self.edge_out_pre = [nn.Sequential(
            nn.Conv2d(inchannels[_], channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.edge_out_pre = nn.ModuleList(self.edge_out_pre)
        self.edge_out = nn.ModuleList([nn.Conv2d(channel, 2, kernel_size=1, bias=False)
                                       for _ in range(self.num_cascade)])
        self.body_out_pre = [nn.Sequential(
            nn.Conv2d(inchannels[_], channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.body_out_pre = nn.ModuleList(self.body_out_pre)
        self.body_out = nn.ModuleList([nn.Conv2d(channel, 2, kernel_size=1, bias=False)
                                       for _ in range(self.num_cascade)])
        self.final_seg_out_pre = [nn.Sequential(
            nn.Conv2d(inchannels[_], channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.final_seg_out_pre = nn.ModuleList(self.final_seg_out_pre)
        self.final_seg_out = nn.ModuleList([nn.Conv2d(channel, 9, kernel_size=1, bias=False)
                                            for _ in range(self.num_cascade)])
    def forward(self, xin, x5,x4,x3,x2):
        x_size=(480 ,640)
        allEncode=[x5,x4,x3,x2]
        seg_edge_outs = []
        seg_body_outs = []
        seg_final_outs = []
        for i in range(self.num_cascade):
            tempX=allEncode[i]+F.interpolate(self.lowGuides[i](x2), size=allEncode[i].size()[2:],
                                              mode='bilinear', align_corners=True)
            if i==0:
                tempX+=xin
            else:
                tempX+=F.interpolate(self.ReduceDemesion[i-1](xin), size=allEncode[i].size()[2:],
                                              mode='bilinear', align_corners=True)
            b,c,h,w=tempX.shape
            tempX=tempX.reshape(b,c,-1).permute(0,2,1)
            tempX=self.NEMs[i](tempX,tempX,tempX)
            tempX = tempX[0].permute(0, 2, 1).reshape(b, c, h,w)
            xin=tempX
            seg_body_out = F.interpolate(self.body_out[i](self.body_out_pre[i](tempX)), size=x_size,
                                         mode='bilinear', align_corners=True)
            seg_body_outs.append(nn.Sigmoid()(seg_body_out))
            seg_edge_out = F.interpolate(self.edge_out[i](self.edge_out_pre[i](tempX)), size=x_size,
                                         mode='bilinear', align_corners=True)
            seg_edge_outs.append(nn.Sigmoid()(seg_edge_out))
            seg_final_out = F.interpolate(self.final_seg_out[i](self.final_seg_out_pre[i](tempX)), size=x_size,
                                          mode='bilinear', align_corners=True)
            seg_final_outs.append(nn.Sigmoid()(seg_final_out))
        return seg_final_outs, seg_body_outs, seg_edge_outs


class DCNet_T(nn.Module):
    def __init__(self, channel=64):
        super(DCNet_T, self).__init__()
        self.rgb = mit_b4()
        self.rgb.init_weights("/home/wby/Desktop/whp_RGBTsemanticsegmentation/toolbox/models/transformer_stack_ModelNet/segformer/pretrained/mit_b2.pth")
        self.thermal = mit_b4()
        self.thermal.init_weights("/home/wby/Desktop/whp_RGBTsemanticsegmentation/toolbox/models/transformer_stack_ModelNet/segformer/pretrained/mit_b2.pth")
        self.dea1 = DEA(64,120,160)
        self.dea2 = DEA(128,60,80)
        self.dea3 = DEA(320,30,40)
        self.dea4 = DEA(512,15,20)
        # Decoder 1
        self.eca = ECA(512)
        self.decoder = Decoder(channel)

    def forward(self, x, x_depth,Aux):
        x1 = self.rgb.forward_featuresOne(x)
        x1_depth = self.thermal.forward_featuresOne(x_depth)
        x1_1 = self.dea1(x1, x1_depth,Aux)
        x2=x1+x1_1
        x2_depth=x1_depth+x1_1
        x2 = self.rgb.forward_featuresTwo(x2)
        x2_depth = self.thermal.forward_featuresTwo(x2_depth)
        x2_1 = self.dea2(x2, x2_depth,Aux)
        x3 = x2 + x2_1
        x3_depth = x2_depth + x2_1
        x3 = self.rgb.forward_featuresThree(x3)
        x3_depth = self.thermal.forward_featuresThree(x3_depth)
        x3_1 = self.dea3(x3, x3_depth,Aux)
        x4 = x3 + x3_1
        x4_depth = x3_depth + x3_1
        x4 = self.rgb.forward_featuresFour(x4)
        x4_depth = self.thermal.forward_featuresFour(x4_depth)
        x4_1 = self.dea4(x4, x4_depth,Aux)
        x4_2 = self.eca(x4_1)
        y = self.decoder(x4_2,x4_1,x3_1,x2_1, x1_1)
        return y

if __name__ == '__main__':
    img = torch.randn(1, 3, 480, 640).cuda()
    depth = torch.randn(1, 3, 480, 640).cuda()
    aux  =  torch.randn(1,3,480, 640).cuda()
    model = DCNet_T().to(torch.device("cuda:0"))
    out = model(img, depth,aux)
    for i in range(len(out[0])):
        print(out[0][i].shape)
