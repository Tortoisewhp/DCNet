import torch.nn.functional as F
import torch
def CL(y_s,y):
    z_s_norm = F.normalize(y_s, dim=1)
    # Compute correlation-matrices
    b,c,h,w=z_s_norm.shape
    z_s_norm=z_s_norm.reshape(b,c,h*w).permute(0,2,1)
    z_s_norm_t=z_s_norm.permute(0,2,1)
    c_ss = torch.bmm(z_s_norm, z_s_norm_t)
    b, c, h, w = y.shape
    y_d = y.reshape(b, c, h * w).permute(0, 2, 1)
    y_d_t = y_d.permute(0, 2, 1)
    yy = torch.bmm(y_d, y_d_t)
    loss=0.0
    loss += torch.log2(c_ss.pow(2).sum()) / (h*h*w*w)
    loss -= torch.log2((c_ss * yy).pow(2).sum()) / (h*h*w*w)
    return loss

def hcl(Ybinary,GTbinary,Ysemantic,GTsemantic):
    return CL(Ybinary,GTbinary)+CL(Ysemantic,GTsemantic)

if __name__ == '__main__':
    Ybinary = torch.randn(2, 1, 30, 40)
    GTbinary = torch.randn(2, 1, 30, 40)
    Ysemantic=torch.randn(2,9,30,40)
    GTsemantic = torch.randn(2,9,30,40)
    print(hcl(Ybinary,GTbinary,Ysemantic,GTsemantic))