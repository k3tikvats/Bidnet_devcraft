import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import sys
import random
import pandas as pd
# sys.path.append('/home/uas-dtu/adobe devcraft/Adobe Devcraft PS')
# from bidder import BidRequest

MODEL_NAME = "sentence-transformers/sentence-t5-xl"
model = SentenceTransformer(MODEL_NAME)


with open(r"profile.json", "r", encoding="utf-8") as f:
    profile_embeddings = json.load(f)

with open(r"avertiser_id.json", "r", encoding="utf-8") as f:
    advertiser_embeddings = json.load(f)


INTERESTS = [
    "10006", "10024", "10031", "10048", "10052", "10057", "10059", "10063", "10067", "10074",
    "10075", "10076", "10077", "10079", "10083", "10093", "10102", "10684", "11092", "11278",
    "11379", "11423", "11512", "11576", "11632", "11680", "11724", "11944", "13042", "13403",
    "13496", "13678", "13776", "13800", "13866", "13874", "14273", "16593", "16617", "16661",
    "16706", "16751", "10110", "10111"
]

def encode_text(sentences):
   
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings

def load_mappings(file_path):
   
    mappings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                mappings[parts[0]] = parts[1] 
    return mappings


city_map = load_mappings(r"city.txt")
region_map = load_mappings(r"region.txt")

# print("City Map Sample:", list(city_map.items())[:5])
# print("Region Map Sample:", list(region_map.items())[:5])

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

def min_max_normalize(values, min_val, max_val):
    """
    Normalize values to the range [0, 1] using min-max normalization.
    """
    return [(x - min_val) / (max_val - min_val) for x in values]

def load_click_data(file_path):
    """
    Load click data and extract BidIDs that received a click.
    """
    clicked_bid_ids = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split("\t")
            bid_id = values[0] 
            clicked_bid_ids.add(bid_id)
    return clicked_bid_ids

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

import torch
from torch.utils.data import Dataset

class RTBDataset(Dataset):
    def __init__(self, imp_file, clk_file, city_file, region_file,limit=2**16):
        self.imp_file = imp_file
        self.clicked_bid_ids = load_click_data(clk_file)
        self.city_map = load_mappings(city_file)
        self.region_map = load_mappings(region_file)
        self.limit=limit
        
        with open(self.imp_file, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        
        self.ip_addresses = [ip_to_numeric(line.strip().split("\t")[5]) for line in self.lines if len(line.strip().split("\t")) >= 24]
        self.min_ip, self.max_ip = min(self.ip_addresses), max(self.ip_addresses)
        np.save('ip_save.npy',np.array([self.min_ip,self.max_ip]))
        
    def __len__(self):
        # return len(self.lines)
        return self.limit

    def __getitem__(self, idx):
        values = self.lines[idx].strip().split("\t")
        if len(values) < 24:
            if len(values)==23:
                interest_scores=torch.zeros(44)
            else:
                return None
        else:
            profile_interests = values[23].split(",")
                # print(values)
            # return __getitem__(self,idx+1)  # Skip invalid rows
        
        bid_id = values[0]
        should_bid = torch.tensor(float(bid_id in self.clicked_bid_ids), dtype=torch.float)
        paying_price = torch.tensor(float(values[20]), dtype=torch.float)
        
        ad_features = torch.tensor([
            int(values[13]), int(values[14]), int(values[16]), int(values[15]), float(values[17])
        ], dtype=torch.float)
        
        normalized_ip = min_max_normalize([ip_to_numeric(values[5])], self.min_ip, self.max_ip)[0]
        city_embedding = encode_text([self.city_map.get(values[7], "unknown")])[0]
        region_embedding = encode_text([self.region_map.get(values[6], "unknown")])[0]
        
        days_since_epoch, time_value = process_timestamp(values[1])
        time_features = torch.tensor([days_since_epoch, time_value, normalized_ip], dtype=torch.float)
        
        
        advertiser_id = values[22]
        if not len(values)<24:
            interest_scores = torch.tensor(get_interest_scores(profile_interests, advertiser_id), dtype=torch.float)
        
        return torch.cat((ad_features, time_features)), city_embedding, region_embedding, interest_scores, torch.tensor([should_bid, paying_price])

def preprocess(b):
    min_ip,max_ip=np.load('Code/ip_save.npy')
    city_file='city.txt'
    region_file='region.txt'
    city_map = load_mappings(city_file)
    region_map = load_mappings(region_file)
    city_embedding=encode_text([city_map.get(b.city,"unknown")])[0]
    region_embedding=encode_text([region_map.get(b.region,"unknown")])[0]
    days_since_epoch, time_value = process_timestamp(b.timestamp)
    normalized_ip = min_max_normalize([ip_to_numeric(b.ipAddress)],min_ip,max_ip)[0]
    time_features = torch.tensor([days_since_epoch, time_value, normalized_ip], dtype=torch.float)
    interest_scores = torch.tensor(get_interest_scores(b.userTags, b.advertiserId), dtype=torch.float)
    ad_features=torch.tensor([float(b.adSlotWidth),float(b.adSlotHeight),float(b.adSlotVisibility),float(b.adSlotFormat),float(b.adSlotFloorPrice)], dtype=torch.float)

    return torch.cat((ad_features, time_features)), city_embedding, region_embedding, interest_scores

def getitem(row,should_bid,paying_price,city_embeddings_dict,id_mapping_dict):
    should_bid = float(should_bid)
    paying_price = float(paying_price) if not pd.isna(paying_price) else 0.0
    advertiser_id = row['AdvertiserID']
    profile_interests = row['User_tag']
    creative_id=row['CreativeID']
    if not pd.isna(profile_interests):
        interest_scores = torch.tensor(get_interest_scores(profile_interests, advertiser_id), dtype=torch.float)
    else:
        interest_scores=torch.zeros(44)
        
    city_embedding = torch.tensor(city_embeddings_dict.get(str(row['City']), city_embeddings_dict["0"]), dtype=torch.float)

    ad_features = torch.tensor([
        int(row['Adslotwidth']), int(row['Adslotheight']),
        int(row['Adslotvisibility']), int(row['Adslotformat']),
        float(row['Adslotfloorprice'])
    ], dtype=torch.float)
    time_features = torch.tensor([row['days_since_epoch'], row['time_part'], row['IP_numeric_normalized']], dtype=torch.float)
    try:
        return (torch.cat((ad_features, time_features,torch.tensor([id_mapping_dict[creative_id]]))).to('cuda'), city_embedding.to('cuda'), interest_scores.to('cuda'))
    except:
        return (torch.cat((ad_features, time_features,torch.tensor([50]))).to('cuda'), city_embedding.to('cuda'), interest_scores.to('cuda'))


class RTBDataset_contrastive(Dataset):
    def __init__(self, imp_file_path, clk_file_path, city_embeddings_path):
        super().__init__()
        with open(city_embeddings_path, 'r') as f:
            self.city_embeddings_dict = json.load(f)
        
        self.column_names = [
        "BidID", "Timestamp", "Logtype", "VisitorID", "User-Agent", "IP", "Region", "City",
        "Adexchange", "Domain", "URL", "AnonymousURLID", "AdslotID", "Adslotwidth",
        "Adslotheight", "Adslotvisibility", "Adslotformat", "Adslotfloorprice",
        "CreativeID", "Biddingprice", "Payingprice", "KeypageURL", "AdvertiserID", "User_tag"
        ]
        self.creative_id_mapping = {'e1af08818a6cd6bbba118bb54a651961': 0, '44966cc8da1ed40c95d59e863c8c75f0': 1, '832b91d59d0cb5731431653204a76c0e': 2, '59f065a795a663140e36eec106464524': 3, '48f2e9ba15708c0146bda5e1dd653caa': 4, 'a499988a822facd86dd0e8e4ffef8532': 5, '4ad7e35171a3d8de73bb862791575f2e': 6, 'b90c12ed2bd7950c6027bf9c6937c48a': 7, '47905feeb59223468fb898b3c9ac024d': 8, '00fccc64a1ee2809348509b7ac2a97a5': 9, 'fe222c13e927077ad3ea087a92c0935c': 10, 'f65c8bdb41e9015970bac52baa813239': 11, '8dff45ed862a740986dbe688aafee7e5': 12, '4b724cd63dfb905ebcd54e64572c646d': 13, 'e049ebe262e20bed5f9b975208db375b': 14, 'cb7c76e7784031272e37af8e7e9b062c': 15, '612599432d200b093719dd1f372f7a30': 16, '23485fcd23122d755d38f8c89d46ca56': 17, '0cd33fcb336655841d3e1441b915748d': 18, '011c1a3d4d3f089a54f9b70a4c0a6eb3': 19, '13606a7c541dcd9ca1948875a760bb31': 20, 'd881a6c788e76c2c27ed1ef04f119544': 21, '80a776343079ed94d424f4607b35fd39': 22, 'd5cecca9a6cbd7a0a48110f1306b26d1': 23, '77819d3e0b3467fe5c7b16d68ad923a1': 24, '2f88fc9cf0141b5bbaf251cab07f4ce7': 25, '86c2543527c86a893d4d4f68810a0416': 26, '3d8f1161832704a1a34e1ccdda11a81e': 27, 'd01411218cc79bc49d2a4078c4093b76': 28, '2abc9eaf57d17a96195af3f63c45dc72': 29, '6b9331e0f0dbbfef42c308333681f0a3': 30, '7eb0065067225fa5f511f7ffb9895f24': 31, '23d6dade7ed21cea308205b37594003e': 32, 'c936045d792f6ea3aa22d86d93f5cf23': 33, 'fb5afa9dba1274beaf3dad86baf97e89': 34, '4400bf8dea968a0068824792fd336c4c': 35, '7097e4210dea4d69f07f0f5e4343529c': 36, '1a43f1ff53f48573803d4a3c31ebc163': 37, '82f125e356439d73902ae85e2be96777': 38, '5c4e0bb0db45e2d1b3a14f817196ebd6': 39, 'ff5123fb9333ca095034c62fdaaf51aa': 40, '62f7f9a6dca2f80cc00f17dcda730bc1': 41, 'c938195f9e404b4f38c7e71bf50263e5': 42, '3b805a00d99d5ee2493c8fb0063e30e9': 43, '87945ed58e806dbdc291b3662f581354': 44, 'e87d7633d474589c2e2e3ba4eda53f6c': 45, '6cdf8fdd3e01122b09b5b411510a2385': 46, '0055e8503dc053435b3599fe44af118b': 47, 'bc27493ad2351e2577bc8664172544f8': 48}
        self.imp = pd.read_csv(imp_file_path, delimiter='\t',names=self.column_names ,low_memory=True)
        self.clk = pd.read_csv(clk_file_path, delimiter='\t',names=self.column_names ,low_memory=True)
        
        self.should_bid = self.imp['BidID'].isin(self.clk['BidID'])

        # Convert to NumPy array for faster indexing
        should_bid_array = self.should_bid.to_numpy()

        # Get indexes where should_bid is 1 (True)
        indexes_1 = np.where(should_bid_array)[0]

        # Get indexes where should_bid is 0 (False)
        indexes_0 = np.where(~should_bid_array)[0]

        self.indexes_1 = indexes_1.tolist()
        self.indexes_0 = indexes_0.tolist()

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
        if(idx in self.indexes_0):
            positive_idx = random.choice(self.indexes_0)
            negative_idx = random.choice(self.indexes_1)
        else:
            positive_idx = random.choice(self.indexes_1)
            negative_idx = random.choice(self.indexes_0)
        
        positive_row = self.imp.iloc[positive_idx]
        negative_row = self.imp.iloc[negative_idx]
            
        anchor = getitem(row,self.should_bid.iloc[idx], self.paying_price.iloc[idx], self.city_embeddings_dict,self.creative_id_mapping)
        positive = getitem(positive_row,self.should_bid.iloc[positive_idx], self.paying_price.iloc[positive_idx], self.city_embeddings_dict,self.creative_id_mapping)
        negative = getitem(negative_row,self.should_bid.iloc[negative_idx], self.paying_price.iloc[negative_idx], self.city_embeddings_dict,self.creative_id_mapping)
    
        return anchor[0],anchor[1],anchor[2], positive[0],positive[1],positive[2],negative[0],negative[1],negative[2]

    
if __name__=='__main__':

    imp_file = r"D:\Adobe Devcraft Dataset\dataset\imp.06.txt"
    clk_file = r"D:\Adobe Devcraft Dataset\dataset\clk.06.txt"
    city_file = r"Adobe Devcraft PS/city.txt"
    region_file = r"Adobe Devcraft PS/region.txt"

    dataset = RTBDataset(imp_file, clk_file, city_file, region_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)


    for batch in dataloader:
        num_features, city_embedding, region_embedding, should_bid, paying_price, interest_scores = zip(*batch)  
        print(f"Numerical Features Shape: {torch.stack(num_features).shape}")  
        print(f"City Embedding Shape (Batch): {[x.shape for x in city_embedding]}")  
        print(f"Region Embedding Shape (Batch): {[x.shape for x in region_embedding]}")
        print(f"Should Bid Shape: {torch.stack(should_bid).shape}") 
        print(f"Paying Price Shape: {torch.stack(paying_price).shape}") 
        print(f"Interest Scores Shape: {torch.stack(interest_scores).shape}") 
        print(interest_scores)
        break 