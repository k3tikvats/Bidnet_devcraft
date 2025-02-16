import torch
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import torch.nn as nn
from model import AutoEncoder
import os 
from new_dataloader import get_interest_scores
import pandas as pd

class AE_Dataset(Dataset):
    def __init__(self,imp_file_path):
        super().__init__()
        self.column_names = [
            "BidID", "Timestamp", "Logtype", "VisitorID", "User-Agent", "IP", "Region", "City",
            "Adexchange", "Domain", "URL", "AnonymousURLID", "AdslotID", "Adslotwidth",
            "Adslotheight", "Adslotvisibility", "Adslotformat", "Adslotfloorprice",
            "CreativeID", "Biddingprice", "Payingprice", "KeypageURL", "AdvertiserID", "User_tag"
            ]
        self.imp = pd.read_csv(imp_file_path, delimiter='\t',names=self.column_names ,low_memory=True)
    
    def __len__(self):
        return len(self.imp)
    
    def __getitem__(self,idx):
        row = self.imp.iloc[idx]
        advertiser_id = row['AdvertiserID']
        profile_interests = row['User_tag']
        if not pd.isna(profile_interests):
            interest_scores = get_interest_scores(profile_interests, advertiser_id)
        else:
            interest_scores=torch.zeros(44)
        return interest_scores

epoch=50
batch=2**15
lr=0.0001
train_file='../../../../ignore/Adobe Devcraft Dataset/dataset/master/train/imp.txt'
test_file='../../../../ignore/Adobe Devcraft Dataset/dataset/master/val/imp.txt'
train_dataset=AE_Dataset(train_file)
train_loader=DataLoader(train_dataset,batch_size=batch,shuffle=True)
test_dataset=AE_Dataset(test_file)
test_loader=DataLoader(test_dataset,batch_size=batch,shuffle=True)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=AutoEncoder()
save_path='model_paramsAutoEncoder'
os.makedirs(save_path)
optimizer=optim.Adam(model.parameters(),lr)
log_interval=5
save_freq=1

for i in epoch:
        epoch_loss=0
        for iter,data in enumerate(train_loader): 
            data=data.to(device)
            predictions=model(data)
            loss=None
            epoch_loss+=loss
            loss.backward()
            if iter%log_interval==0:
                print(f'epoch:{i}/{epoch} iteration:{iter}/{len(train_dataset)//batch+1} batch loss is :{loss:.4f}')
            optimizer.step()
            optimizer.zero_grad()
            if iter%20==0:
                torch.save(model.state_dict(),f'{save_path}/epoch{i}_iter_{iter}_train_{epoch_loss}')
        epoch_loss/=(len(train_dataset)//batch+1)
        print(f'Epoch:{i}/{epoch} Loss is :{epoch_loss:.4f}')
        print("average epoch loss':f'{epoch_loss:.4f}")
        
        if i%save_freq==0:
            val_loss=0
            for data in test_loader:
                data=data.to(device)
                val_predictions=model(data)
                
                loss=None
                val_loss+=loss
            val_loss/=(len(test_dataset)//batch+1)
            print(f'VAL loss :{val_loss:.4f}')
            
            torch.save(model.state_dict(),f'{save_path}/epoch{i}_val_{val_loss}_train_{epoch_loss}')
            