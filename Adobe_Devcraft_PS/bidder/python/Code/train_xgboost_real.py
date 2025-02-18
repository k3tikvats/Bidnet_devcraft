import joblib
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from datetime import datetime
import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from tqdm import tqdm
from model import BidPredictor

REFERENCE_DATE = datetime(1970, 1, 1)

def process_timestamp(timestamp):
   
    date_part = timestamp[:8]  
    time_part = timestamp[8:] 

    date_obj = datetime.strptime(date_part, "%Y%m%d")
    days_since_epoch = (date_obj - REFERENCE_DATE).days

    return days_since_epoch, int(time_part)

def replace_x_with_127(ip_address):
   
    if ip_address.endswith(".*"):
        return ip_address[:-2] + ".127"
    return ip_address

def ip_to_numeric(ip_address):
    """
    Convert an IP address (e.g., '192.168.1.1') to a 32-bit integer.
    If the IP address is invalid or missing, return a default value (e.g., 127).
    """
    try:
        
        ip_address = replace_x_with_127(ip_address)
        octets = list(map(int, ip_address.split(".")))
        numeric_ip = (octets[0] << 24) + (octets[1] << 16) + (octets[2] << 8) + octets[3]
        return numeric_ip
    except (IndexError, ValueError, AttributeError):
        return 127

profile_json_path = f"../profile.json"
with open(profile_json_path, "r", encoding="utf-8") as f:
    profile_embeddings = json.load(f)

advertiser_json_path = f"../avertiser_id.json"
with open(advertiser_json_path, "r", encoding="utf-8") as f:
    advertiser_embeddings = json.load(f)
    
INTERESTS = [
    "10006", "10024", "10031", "10048", "10052", "10057", "10059", "10063", "10067", "10074",
    "10075", "10076", "10077", "10079", "10083", "10093", "10102", "10684", "11092", "11278",
    "11379", "11423", "11512", "11576", "11632", "11680", "11724", "11944", "13042", "13403",
    "13496", "13678", "13776", "13800", "13866", "13874", "14273", "16593", "16617", "16661",
    "16706", "16751", "10110", "10111"
]

def get_interest_scores(profile_interests, advertiser_id):
    interest_vector = np.zeros(44)  
    advertiser_data = advertiser_embeddings.get(str(advertiser_id), {})
    advertiser_embedding = np.array(advertiser_data.get("embed", [0] * 512))
    n_value = float(advertiser_data.get("N", 0))
    for i, interest_id in enumerate(INTERESTS):
        if interest_id in profile_interests:
            profile_embedding = np.array(profile_embeddings.get(interest_id, {}).get("embed", [0] * 512))
            similarity = cosine_similarity([advertiser_embedding], [profile_embedding])[0][0]
            interest_vector[i] = similarity*n_value 
    return interest_vector

class RTBDataset_new(Dataset):
    def __init__(self, imp_file_path, clk_file_path, city_embeddings_path):
        super().__init__()
        with open(city_embeddings_path, 'r') as f:
            self.city_embeddings_dict = json.load(f)

        self.creative_id_mapping = {'e1af08818a6cd6bbba118bb54a651961': 0, '44966cc8da1ed40c95d59e863c8c75f0': 1, '832b91d59d0cb5731431653204a76c0e': 2, '59f065a795a663140e36eec106464524': 3, '48f2e9ba15708c0146bda5e1dd653caa': 4, 'a499988a822facd86dd0e8e4ffef8532': 5, '4ad7e35171a3d8de73bb862791575f2e': 6, 'b90c12ed2bd7950c6027bf9c6937c48a': 7, '47905feeb59223468fb898b3c9ac024d': 8, '00fccc64a1ee2809348509b7ac2a97a5': 9, 'fe222c13e927077ad3ea087a92c0935c': 10, 'f65c8bdb41e9015970bac52baa813239': 11, '8dff45ed862a740986dbe688aafee7e5': 12, '4b724cd63dfb905ebcd54e64572c646d': 13, 'e049ebe262e20bed5f9b975208db375b': 14, 'cb7c76e7784031272e37af8e7e9b062c': 15, '612599432d200b093719dd1f372f7a30': 16, '23485fcd23122d755d38f8c89d46ca56': 17, '0cd33fcb336655841d3e1441b915748d': 18, '011c1a3d4d3f089a54f9b70a4c0a6eb3': 19, '13606a7c541dcd9ca1948875a760bb31': 20, 'd881a6c788e76c2c27ed1ef04f119544': 21, '80a776343079ed94d424f4607b35fd39': 22, 'd5cecca9a6cbd7a0a48110f1306b26d1': 23, '77819d3e0b3467fe5c7b16d68ad923a1': 24, '2f88fc9cf0141b5bbaf251cab07f4ce7': 25, '86c2543527c86a893d4d4f68810a0416': 26, '3d8f1161832704a1a34e1ccdda11a81e': 27, 'd01411218cc79bc49d2a4078c4093b76': 28, '2abc9eaf57d17a96195af3f63c45dc72': 29, '6b9331e0f0dbbfef42c308333681f0a3': 30, '7eb0065067225fa5f511f7ffb9895f24': 31, '23d6dade7ed21cea308205b37594003e': 32, 'c936045d792f6ea3aa22d86d93f5cf23': 33, 'fb5afa9dba1274beaf3dad86baf97e89': 34, '4400bf8dea968a0068824792fd336c4c': 35, '7097e4210dea4d69f07f0f5e4343529c': 36, '1a43f1ff53f48573803d4a3c31ebc163': 37, '82f125e356439d73902ae85e2be96777': 38, '5c4e0bb0db45e2d1b3a14f817196ebd6': 39, 'ff5123fb9333ca095034c62fdaaf51aa': 40, '62f7f9a6dca2f80cc00f17dcda730bc1': 41, 'c938195f9e404b4f38c7e71bf50263e5': 42, '3b805a00d99d5ee2493c8fb0063e30e9': 43, '87945ed58e806dbdc291b3662f581354': 44, 'e87d7633d474589c2e2e3ba4eda53f6c': 45, '6cdf8fdd3e01122b09b5b411510a2385': 46, '0055e8503dc053435b3599fe44af118b': 47, 'bc27493ad2351e2577bc8664172544f8': 48}

        
        self.column_names = [
        "BidID", "Timestamp", "Logtype", "VisitorID", "User-Agent", "IP", "Region", "City",
        "Adexchange", "Domain", "URL", "AnonymousURLID", "AdslotID", "Adslotwidth",
        "Adslotheight", "Adslotvisibility", "Adslotformat", "Adslotfloorprice",
        "CreativeID", "Biddingprice", "Payingprice", "KeypageURL", "AdvertiserID", "User_tag"
        ]
        
        self.imp = pd.read_csv(imp_file_path, delimiter='\t',names=self.column_names ,low_memory=True)
        self.clk = pd.read_csv(clk_file_path, delimiter='\t',names=self.column_names ,low_memory=True)
        
        self.should_bid = self.imp['BidID'].isin(self.clk['BidID'])
        self.paying_price = self.imp['Payingprice']
        
        self.imp['IP_numeric'] = self.imp['IP'].apply(ip_to_numeric)
        min_value = self.imp['IP_numeric'].min()
        max_value = self.imp['IP_numeric'].max()
        self.imp['IP_numeric_normalized'] = (self.imp['IP_numeric'] - min_value) / (max_value - min_value)
        self.imp[['days_since_epoch', 'time_part']] = self.imp['Timestamp'].apply(
            lambda x: pd.Series(process_timestamp(str(x)))
        )
        
    def __len__(self):
        return len(self.imp)

    def __getitem__(self, idx):
        row = self.imp.iloc[idx]
        should_bid = float(self.should_bid.iloc[idx])
        paying_price = float(self.paying_price.iloc[idx]) if not pd.isna(self.paying_price.iloc[idx]) else 0.0
        advertiser_id = row['AdvertiserID']
        profile_interests = row['User_tag']
        if not pd.isna(profile_interests):
            interest_scores = get_interest_scores(profile_interests, advertiser_id)
        else:
            interest_scores=torch.zeros(44)
            
        city_embedding = self.city_embeddings_dict.get(str(row['City']), self.city_embeddings_dict["0"])
        try:
            ad_features = np.array([
                int(row['Adslotwidth']), int(row['Adslotheight']),
                int(row['Adslotvisibility']), int(row['Adslotformat']),
                float(row['Adslotfloorprice']),row['days_since_epoch'], row['time_part'], row['IP_numeric_normalized'],
                int(self.creative_id_mapping[row['CreativeID']])
            ])
        except:
            ad_features = np.array([
                int(row['Adslotwidth']), int(row['Adslotheight']),
                int(row['Adslotvisibility']), int(row['Adslotformat']),
                float(row['Adslotfloorprice']),row['days_since_epoch'], row['time_part'], row['IP_numeric_normalized'],
                int(50)
            ])
    
       
        # features = np.concatenate([
        #     ad_features,
        #     city_embedding,
        #     interest_scores
        # ])
        return torch.tensor(ad_features),torch.tensor(city_embedding),torch.tensor(interest_scores) , should_bid, paying_price
    
if __name__ == "__main__":

    imp_file = r"/media/uas-dtu/OLDUBNT/random/adobe devcraft (1)/ignore/Adobe Devcraft Dataset/dataset/master/val/imp.txt"
    clk_file = r"/media/uas-dtu/OLDUBNT/random/adobe devcraft (1)/ignore/Adobe Devcraft Dataset/dataset/merged_clk.txt"
    city_file = r"city_embeddings.json"
    
    print('intialising dataset.....')
    dataset = RTBDataset_new(imp_file, clk_file, city_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print('initalisation complete!')

    all_features = []
    all_should_bid = []
    all_paying_price = []
    
    model_weights='model_paramsRepresentative/epoch16_iter_220_train_44.93674087524414'
    model=BidPredictor(768).to('cuda')
    model.load_state_dict(torch.load(model_weights),strict=False)

    for batch in tqdm(dataloader):
        # print("in loop")
        ad_features,city,interest, should_bid, paying_price = batch
        all_features.append(model(ad_features.to('cuda').float(),city.to('cuda').float(),interest.to('cuda').float()).detach().cpu().numpy( ))
        all_should_bid.append(should_bid)
        all_paying_price.append(paying_price)

    X = np.vstack(all_features)
    y_bid = np.hstack(all_should_bid)
    y_price = np.hstack(all_paying_price)

    X_train, X_test, y_bid_train, y_bid_test, y_price_train, y_price_test = train_test_split(
        X, y_bid, y_price, test_size=0.05, random_state=42
    )
    print(X.shape,y_price.shape,y_bid_train.shape)
    
    scale_pos_weight = (y_bid_train == 0).sum() / (y_bid_train == 1).sum()
  
    xgb_params_bid = {
        "n_estimators": 100, 
        "max_depth": 5,       
        "learning_rate": 0.1,  
        "objective": "binary:logistic", 
        "eval_metric": "logloss",  
        "use_label_encoder": False,
        "subsample": 0.8,    
        "colsample_bytree": 0.8, 
        "reg_alpha": 0.1,    
        "reg_lambda": 0.1,    
        "seed": 42,
        "scale_pos_weight": scale_pos_weight           
    }
    quantile_alpha = 0.1
    xgb_params_price = {
        "n_estimators": 100,  
        "max_depth": 5,      
        "learning_rate": 0.1, 
        "objective": "reg:quantileerror",  
        "quantile_alpha": quantile_alpha, 
        "eval_metric": "rmse", 
        "subsample": 0.8,    
        "colsample_bytree": 0.8, 
        "reg_alpha": 0.1,     
        "reg_lambda": 0.1,    
        "seed": 42            
    }

    print('training_started')
    model_bid = XGBClassifier(**xgb_params_bid)
    model_bid.fit(X_train, y_bid_train)

    model_filename = 'xgb_bid_model.joblib'
    joblib.dump(model_bid, model_filename)

    print(f"Model saved to {model_filename}")

   
    y_bid_pred = model_bid.predict(X_test)
    accuracy = accuracy_score(y_bid_test, y_bid_pred)
    print(f"Bidding Accuracy: {accuracy:.4f}")

    model_price = XGBRegressor(**xgb_params_price)
    model_price.fit(X_train[y_bid_train==1,:], y_price_train[y_bid_train==1])

    model_filename = 'xgb_price_model.joblib'
    joblib.dump(model_price, model_filename)

    print(f"Model saved to {model_filename}")

    y_price_pred = model_price.predict(X_test)
    rmse = mean_squared_error(y_price_test, y_price_pred)
    print(f"Paying Price RMSE: {rmse:.4f}")

    np.save('eval.npy',np.array([accuracy,rmse]))