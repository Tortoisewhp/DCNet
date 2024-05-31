import torch
import torch.nn as nn
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

def SoftenSupervisionCondition1(G,GT):
    GT[GT>torch.tensor(0.5)]=torch.tensor(1.0).cuda()
    GT[GT<torch.tensor(0.5)]=torch.tensor(0.0).cuda()
    G[torch.logical_and(G*GT<torch.tensor(0.5),G+GT>=torch.tensor(1.0))]=G[torch.logical_and(G*GT<torch.tensor(0.5),G+GT>=torch.tensor(1.0))]+torch.tensor(0.5).cuda()
    G[torch.logical_and(torch.logical_and(G * GT < torch.tensor(0.5),G + GT <= torch.tensor(1.0)),G>torch.tensor(0.5))] = G[torch.logical_and(torch.logical_and(G * GT < torch.tensor(0.5),G + GT <= torch.tensor(1.0)),G>torch.tensor(0.5))]  - torch.tensor(0.5).cuda()
    return G

def spatialRelation(featureX,featureY):
    b,c,h,w=featureX.shape
    feature1=featureX.reshape(b,c,-1)
    feature2=featureY.reshape(b,h*w,-1)
    out=torch.bmm(feature2,feature1)
    out=torch.softmax(out,2)
    return out

def SRRD(ySemantic,YSemantic,GT):
    y=SoftenSupervisionCondition1(ySemantic,GT)
    Y=SoftenSupervisionCondition1(YSemantic,GT)
    ry=spatialRelation(y,y)
    rY=spatialRelation(Y,Y)
    loss=nn.CrossEntropyLoss()(ry,rY)
    return loss

if __name__ == '__main__':
    yFusedFeature=torch.randn(2,9,30,40).cuda()
    YFusedFeature = torch.randn(2,9,30,40).cuda()
    GT=torch.randint(0,2,(2,9,30,40),dtype=torch.float32).cuda()
    print(SRRD(yFusedFeature,YFusedFeature,GT))

