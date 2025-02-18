import torch
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import torch.nn as nn
from model import AutoEncoder
import os 
from new_dataloader import get_interest_scores
import pandas as pd
from loss import LogScaledLoss
from tqdm import tqdm
import json
class AE_Dataset(Dataset):
    def __init__(self,imp_file_path,city_embeddings_path):
        super().__init__()
        with open(city_embeddings_path, 'r') as f:
            self.city_embeddings_dict = json.load(f)
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
        city_embedding = self.city_embeddings_dict.get(str(row['City']), self.city_embeddings_dict["0"])

        return torch.tensor(city_embedding)

epoch=50
batch=2**10
lr=0.0001
train_file='../../../../ignore/Adobe Devcraft Dataset/dataset/val/val06.txt'
test_file='../../../../ignore/Adobe Devcraft Dataset/dataset/val/val07.txt'
embedding_json='city_embeddings.json'
train_dataset=AE_Dataset(train_file,embedding_json)
train_loader=DataLoader(train_dataset,batch_size=batch,shuffle=True)
test_dataset=AE_Dataset(test_file,embedding_json)
test_loader=DataLoader(test_dataset,batch_size=batch,shuffle=True)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=AutoEncoder().to(device)
save_path='model_paramsAutoEncoder'
os.makedirs(save_path,exist_ok=True)
optimizer=optim.Adam(model.parameters(),lr)
log_interval=5
save_freq=5
# criteron=LogScaledLoss()
criteron=nn.MSELoss()

pbar=tqdm(range(epoch))
for i in pbar:
        epoch_loss=0
        for iter,data in enumerate(train_loader): 
            data=data.to(device).float()
            # data=(data+11)/22#normalisation
            predictions=model(data)
            loss=criteron(data,predictions)
            epoch_loss+=loss
            loss.backward()
            if iter%log_interval==0:
                # print(f'epoch:{i}/{epoch} iteration:{iter}/{len(train_dataset)//batch+1} batch loss is :{loss:.4f}')
                pbar.set_postfix({'epoch':str(i),'iteration':str(iter),'loss':str(loss)})
            optimizer.step()
            optimizer.zero_grad()
            if iter%20==0:
                torch.save(model.state_dict(),f'{save_path}/epoch{i}_iter_{iter}_train_{epoch_loss}')
        epoch_loss/=(len(train_dataset)//batch+1)
        # print(f'Epoch:{i}/{epoch} Loss is :{epoch_loss:.4f}')
        # print("average epoch loss':f'{epoch_loss:.4f}")
        pbar.set_postfix({'epoch':str(i),'epoch_loss':str(epoch_loss)})
        
        if i%save_freq==0:
            val_loss=0
            for data in test_loader:
                data=data.to(device)
                val_predictions=model(data)               
                loss=criteron(data,val_predictions)
                val_loss+=loss
            val_loss/=(len(test_dataset)//batch+1)
            print(f'VAL loss :{val_loss:.4f}')
            
            torch.save(model.state_dict(),f'{save_path}/epoch{i}_val_{val_loss}_train_{epoch_loss}')

            