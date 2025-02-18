# %%
import torch
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import torch.nn as nn
from model import BidPredictor
import os 
from new_dataloader import get_interest_scores
import pandas as pd
from loss import LogScaledLoss
from tqdm import tqdm
import json
from dataloader_old import RTBDataset_contrastive
from torchsummary import summary
epoch=50
batch=2**10
lr=0.0001
train_imp_file='../../../../ignore/Adobe Devcraft Dataset/dataset/master/val/imp.txt'
test_imp_file='../../../../ignore/Adobe Devcraft Dataset/dataset/val/val10.txt'
clk_file='../../../../ignore/Adobe Devcraft Dataset/dataset/master/clk.txt'
embedding_json='city_embeddings.json'

train_dataset=RTBDataset_contrastive(train_imp_file,clk_file,embedding_json)
train_loader=DataLoader(train_dataset,batch_size=batch,shuffle=True)
test_dataset=RTBDataset_contrastive(test_imp_file,clk_file,embedding_json)
test_loader=DataLoader(test_dataset,batch_size=batch,shuffle=True)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=BidPredictor(768).to(device)
summary(model,[(1,9),(1,768),(1,44)],batch_size=1) 
save_path='model_paramsRepresentative'
os.makedirs(save_path,exist_ok=True)
optimizer=optim.Adam(model.parameters(),lr)
log_interval=5
save_freq=5
# criteron=LogScaledLoss()
criteron=nn.TripletMarginLoss()

pbar=tqdm(range(epoch))

for i in pbar:
        epoch_loss=0
        for iter,(anchor1,anchor2,anchor3,positive1,positive2,positive3,negative1,negative2,negative3) in enumerate(train_loader): 
            # anchor=[val.to(device).float() for val in anchor]
            # positive=[val.to(device).float() for val in positive]
            # negative=[val.to(device).float() for val in negative]

            predictions_anchor=model(anchor1,anchor2,anchor3)
            predictions_positive=model(positive1,positive2,positive3)
            predictions_negative=model(negative1,negative2,negative3)
            loss=criteron(predictions_anchor,predictions_positive,predictions_negative)
            epoch_loss+=loss
            loss.backward()
            if iter%log_interval==0:

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
            for (anchor1,anchor2,anchor3,positive1,positive2,positive3,negative1,negative2,negative3)  in test_loader:

                val_predictions_anchor=model(anchor1,anchor2,anchor3)
                val_predictions_positive=model(positive1,positive2,positive3)
                val_predictions_negative=model(negative1,negative2,negative3)
                loss=criteron(val_predictions_anchor,val_predictions_positive,val_predictions_negative)
                val_loss+=loss
            val_loss/=(len(test_dataset)//batch+1)
            print(f'VAL loss :{val_loss:.4f}')
            
            torch.save(model.state_dict(),f'{save_path}/epoch{i}_val_{val_loss}_train_{epoch_loss}')
            