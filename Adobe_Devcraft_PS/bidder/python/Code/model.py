import torch.nn as nn
import torch
class Residual(nn.Module):
    def __init__(self,layers:list):
        super().__init__()
        self.layers=layers
        assert len(self.layers)==3,f'the number of layers should be 3 not {len(self.layers)}'
        self.silu=nn.SiLU()
        self.normalise1=nn.BatchNorm1d(layers[1])
        self.normalise2=nn.BatchNorm1d(layers[2])
        self.layer1=nn.Linear(self.layers[0],self.layers[1])
        self.layer2=nn.Linear(self.layers[1],self.layers[2])
        self.project=nn.Linear(self.layers[0],self.layers[2])

    def forward(self,x):
        # print(self.layers)
        x_=self.normalise1(self.silu(self.layer1(x)))
        # print('fuck off')
        # print(x_.shape)
        # print(self.layers[1],self.layers[2])
        x1=self.layer2(x_)
        # +self.project(x)
        return self.normalise2(self.silu(x1))
class BidPredictor(nn.Module):
    def __init__(self,loc_tokenizer_latent_dim,compression=5,time_features=2):
        super().__init__()
        self.loc_compression=[loc_tokenizer_latent_dim//(4)**i for i in range(compression)]
        # self.num_features=10+time_features+2*self.loc_compression[-1]
        self.num_features=16+3+9
        # self.Region_compressor=nn.Sequential(*[self.block(self.loc_compression[i],self.loc_compression[i+1]) for i in range(compression-1)])
        # self.Region_compressor=nn.Sequential(*self.make_compressor(self.loc_compression))
        # self.City_compressor=nn.Sequential(*[self.block(self.loc_compression[i],self.loc_compression[i+1]) for i in range(compression-1)])
        # self.City_compressor=nn.Sequential(*self.make_compressor(self.loc_compression))
        self.City_compressor=AutoEncoder(dim=loc_tokenizer_latent_dim)
        alignment_layers=[44,32,16]
        self.alignment=AutoEncoder(layers=alignment_layers)
        # self.alignment=nn.Sequential(*[self.block(alignment_layers[i],alignment_layers[i+1]) for i in range(len(alignment_layers)-1)])
        # self.backbone_layers=[self.num_features,64,64,128,256,512]
        self.backbone_layers=[self.num_features,64,128]
        # self.backbone=nn.Sequential(*[self.block(self.backbone_layers[i],self.backbone_layers[i+1]) for i in range(len(self.backbone_layers)-1)])
        self.backbone=nn.Sequential(*self.make_compressor(self.backbone_layers))
        self.classifier=nn.Linear(self.backbone_layers[-1],2)
        # self.sigmoid=nn.Sigmoid()
        
    def make_compressor(self,channels):
        num_layers=len(channels)
        layers=[]
        print(channels)
        for i in range(0,num_layers-2,2):
            layers.append(Residual([channels[i],channels[i+1],channels[i+2]]))
        if (i%2!=0):
            layers.append(nn.Sequential(nn.Linear(channels[-2],channels[-1]),nn.SiLU()))
        return layers

    @staticmethod
    def block(in_feature,out_feature):
        return nn.Sequential(nn.Linear(in_feature,out_feature),nn.BatchNorm1d(out_feature),nn.SiLU())
    
    def forward(self,x,City,alignment):
        City=City.reshape([-1,768])
        alignment=alignment.reshape([-1,44])
        x=x.reshape([-1,9])
        city_features=self.City_compressor.latent(City)
        alignment=self.alignment.latent(alignment)
        x=torch.concat([x,city_features,alignment],dim=-1)
        assert x.shape[-1]==self.num_features,f'the number of features should be {self.num_features}'
        out=self.backbone(x)
        return self.classifier(out)

class AutoEncoder(nn.Module):
    def __init__(self,dim=None,layers=None):
        super().__init__()
        # layers=[44,32,16]
        assert not dim or not layers,'incomplete data'

        if not layers:           
            layers=[dim//(4)**i for i in range(5)]

        self.encoder=nn.Sequential(*self.make_compressor(layers))
        self.bottleneck=nn.Sequential(*[nn.Linear(layers[-1],layers[-1]),nn.SiLU()])
        self.decoder=nn.Sequential(*self.make_compressor([layers[-(i+1)] for i in range(len(layers))]))

    def make_compressor(self,channels):
        num_layers=len(channels)
        layers=[]
        print(channels)
        for i in range(0,num_layers-2,2):
            layers.append(Residual([channels[i],channels[i+1],channels[i+2]]))
        if (i%2!=0):
            layers.append(nn.Sequential(nn.Linear(channels[-2],channels[-1]),nn.SiLU()))
        return layers

    @staticmethod
    def block(in_feature,out_feature):
        return nn.Sequential(nn.Linear(in_feature,out_feature),nn.BatchNorm1d(out_feature),nn.SiLU())
    
    def forward(self,x):
        x=x.reshape([-1,768])
        x=self.encoder(x)
        x=self.bottleneck(x)
        x=self.decoder(x)
        return x
    def latent(self,x):
        x=self.encoder(x)
        x=self.bottleneck(x)
        return x
if __name__=='__main__':
    from torchsummary import summary
    model=BidPredictor(768).to('cuda')
    model(torch.ones((2,9),device='cuda'),torch.ones((2,768),device='cuda'),torch.ones((2,44),device='cuda'))
    summary(model,[(1,9),(1,768),(1,44)],batch_size=1)        
    # model=AutoEncoder().to('cuda')
    # model(torch.ones((2,768),device='cuda'))
    # summary(model,(1,768),batch_size=2)