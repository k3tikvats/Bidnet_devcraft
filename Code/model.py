import torch.nn as nn
import torch

class BidPredictor(nn.Module):
    def __init__(self,loc_tokenizer_latent_dim,compression=5,time_features=2):
        super().__init__()
        self.loc_compression=[loc_tokenizer_latent_dim//(4)**i for i in range(compression)]
        self.num_features=10+time_features+2*self.loc_compression[-1]    
        self.Region_compressor=nn.Sequential(*[self.block(self.loc_compression[i],self.loc_compression[i+1]) for i in range(compression-1)])
        self.City_compressor=nn.Sequential(*[self.block(self.loc_compression[i],self.loc_compression[i+1]) for i in range(compression-1)])
        alignment_layers=[44,8,4]
        self.alignment=nn.Sequential(*[self.block(alignment_layers[i],alignment_layers[i+1]) for i in range(len(alignment_layers)-1)])
        self.backbone_layers=[self.num_features,64,64,128,256,512]
        self.backbone=nn.Sequential(*[self.block(self.backbone_layers[i],self.backbone_layers[i+1]) for i in range(len(self.backbone_layers)-1)])
        self.classifier=nn.Linear(self.backbone_layers[-1],2)
        # self.sigmoid=nn.Sigmoid()

    @staticmethod
    def block(in_feature,out_feature):
        return nn.Sequential(nn.Linear(in_feature,out_feature),nn.BatchNorm1d(out_feature),nn.SiLU())
    
    def forward(self,x,Region,City,alignment):
        # x=torch.reshape(x,(-1,9))
        # Region=torch.reshape(Region,(-1,512))
        # City=torch.reshape(City,(-1,512))
        # print(x.shape)
        region_features=self.Region_compressor(Region)
        city_features=self.City_compressor(City)
        alignment=self.alignment(alignment)
        x=torch.concat([x,region_features,city_features,alignment],dim=-1)
        assert x.shape[-1]==self.num_features,f'the number of features should be {self.num_features}'
        out=self.backbone(x)
        return self.classifier(out)
 
if __name__=='__main__':
    from torchsummary import summary
    model=BidPredictor(512)
    # model(torch.ones((2,9)),torch.ones(2,512),torch.ones(2,512))
    summary(model,[(2,9),(2,512),(2,512)],batch_size=1)        