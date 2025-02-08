import argparse
import torchvision
import torch
from torchvision.transforms import transforms
import torch.optim as optim
from loss import QuantileLoss
from model import BidPredictor
import os
from tqdm import tqdm
import logging
import torch.nn as nn

parser = argparse.ArgumentParser(description="train arguments")



parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument('--batch',type=float,default=32,help='batch size')
parser.add_argument('--epoch',type=int,default=50,help='number of epoch')
parser.add_argument('--data_dir',type=str,default='',help="path of data")
parser.add_argument('--save_dir',type=str,default='model_parameter')
parser.add_argument('--weight_fac',type=float,default=0.3,help='weight factor of q loss')
parser.add_argument('--save_freq',type=int,default=2,help='after how many epochs are the parameters saved')
parser.add_argument('--log_dir',type=str,default='logs',help='the directory in which training logs are to be saved')
parser.add_argument('--gamma',type=float,default=0.1,help='gamma for learning rate decay')
parser.add_argument('--step_size',type=int,default=2,help='number of epochs after which learning rate is to be decayed')
parser.add_argument('--model_path',type=str,default=None,help='path of model parameters to be loaded')
parser.add_argument('--log_interval',type=int,default=100,help='number of iterations after with loss is logged')
parser.add_argument('--log_interval',type=int,default=100,help='number of iterations after with loss is logged')
parser.add_argument('--embed_dim',type=int,default=None,help='latent dim of sentencepiece tokenizer')
args = parser.parse_args()

if os.path.exists(args.save_dir):
    num_folder=str(int(sorted(os.listdir(args.save_dir),key=lambda x:int(x))[-1])+1)
else:
    os.makedirs(args.save_dir,exist_ok=False)
    num_folder=0
save_path=args.save_dir+os.sep+num_folder

os.makedirs(args.log_dir,exist_ok=True)
logging.basicConfig(
    level=logging.INFO,                  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S',        
    handlers=[
        # logging.StreamHandler(),        
        logging.FileHandler(args.log_dir+os.sep+f"train_{num_folder}.log")  
    ]
)
logging.info('------------------------------------------------------------------------------------------------')
logging.info('starting new training !!!!')
logging.info('------------------------------------------------------------------------------------------------')

logging.info(args)

    
os.makedirs(save_path)
logging.info(f'parameters are being saved at :{save_path}')
model=BidPredictor(512)
logging.info(model)
quantile_criteron=QuantileLoss()
BCE_criteron=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),args.lr)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

if args.model_path:
    model.load_state_dict(torch.load(args.model_path))
    print('loaded model parameters!')

train_dataset=None
train_loader=None
test_dataset=None
test_loader=None

pbar=tqdm(range(args.epoch))
for i in pbar:
        epoch_loss=0
        epoch_cls_loss=0
        epoch_q_loss=0
        for iter,(data,city,region,label) in enumerate(train_loader):
            data=data.to(device)
            city=city.to(device)
            region=region.to(device)
            label=label.to(device)
            predictions=model(data,city,region)
            q_loss=quantile_criteron(label[:,1],predictions[:,1])
            cls_loss=BCE_criteron(predictions[:,0],label[:,0])

            loss=cls_loss+args.weight_fac*q_loss
            epoch_loss+=loss
            epoch_cls_loss+=cls_loss
            epoch_q_loss+=q_loss
            if iter%args.log_interval==0:
                logging.info(f'epoch:{i}/{args.epoch} iteration:{iter}/{len(train_dataset)//args.batch+1} batch loss is :{loss:.4f}')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss/=(len(train_dataset)//args.batch+1)
        epoch_cls_loss/=(len(train_dataset)//args.batch+1)
        epoch_q_loss/=(len(train_dataset)//args.batch+1)
        logging.info(f'Epoch:{i}/{args.epoch} Loss is :{epoch_loss:.4f}')
        logging.info(f'average cls loss is :{epoch_cls_loss:.4f}')
        logging.info(f'average q divergence is : {epoch_q_loss:.4f}')
        pbar.set_postfix({'average epoch loss':f'{epoch_loss:.4f}','cls loss':f'{epoch_cls_loss:.4f}','q_loss':f'{epoch_q_loss:.4f}'})
        # print(f'Epoch:{i}/{args.epoch} Loss is :{epoch_loss:.4f}')
        scheduler.step()
        if i%args.save_freq==0:
            val_loss=0
            for data,city,region,label in test_loader:
                data=data.to(device)
                city=city.to(device)
                region=region.to(device)
                label=label.to(device)
                val_predictions=model(data,city,region)
                q_loss=quantile_criteron(label[:,1],val_predictions[:,1])
                cls_loss=BCE_criteron(val_predictions[:,0],label[:,0])

                loss=cls_loss+args.weight_fac*q_loss
                val_loss+=loss
            val_loss/=(len(test_dataset)//args.batch+1)
            logging.info(f'VAL loss :{val_loss:.4f}')
            
            torch.save(model.state_dict(),f'{save_path}/epoch{i}_val_{val_loss}_train_{epoch_loss}')
            
print('training completed !!')
logging.info('training complete!!!')