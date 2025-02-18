import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
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
import matplotlib.pyplot as plt
import seaborn as sns

REFERENCE_DATE = datetime(2012, 1, 1)

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

profile_json_path = r"profile.json"
with open(profile_json_path, "r", encoding="utf-8") as f:
    profile_embeddings = json.load(f)

advertiser_json_path = r"avertiser_id.json"
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
    return interest_vector# ... (rest of your code remains the same)

class RTBDataset_new(Dataset):
    def __init__(self, imp_file_path, clk_file_path, city_embeddings_path, region_embeddings_path):
        super().__init__()
        with open(city_embeddings_path, 'r') as f:
            self.city_embeddings_dict = json.load(f)

        with open(region_embeddings_path, 'r') as f:
            self.region_embeddings_dict = json.load(f)
        
        self.column_names = [
            "BidID", "Timestamp", "Logtype", "VisitorID", "User-Agent", "IP", "Region", "City",
            "Adexchange", "Domain", "URL", "AnonymousURLID", "AdslotID", "Adslotwidth",
            "Adslotheight", "Adslotvisibility", "Adslotformat", "Adslotfloorprice",
            "CreativeID", "Biddingprice", "Payingprice", "KeypageURL", "AdvertiserID", "User_tag"
        ]
        
        self.imp = pd.read_csv(imp_file_path, delimiter='\t', names=self.column_names, low_memory=True)
        self.clk = pd.read_csv(clk_file_path, delimiter='\t', names=self.column_names, low_memory=True)
        
        self.should_bid = self.imp['BidID'].isin(self.clk['BidID'])
        self.paying_price = self.imp['Payingprice']
        
        self.imp['IP_numeric'] = self.imp['IP'].apply(ip_to_numeric)
        min_value = self.imp['IP_numeric'].min()
        max_value = self.imp['IP_numeric'].max()
        self.imp['IP_numeric_normalized'] = (self.imp['IP_numeric'] - min_value) / (max_value - min_value)
        self.imp[['days_since_epoch', 'time_part']] = self.imp['Timestamp'].apply(lambda x: pd.Series(process_timestamp(str(x))))
        
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
            interest_scores = torch.zeros(44)
            
        city_embedding = self.city_embeddings_dict.get(str(row['City']), self.city_embeddings_dict["0"])
        region_embedding = self.region_embeddings_dict.get(str(row['Region']), self.region_embeddings_dict["0"])
        
        # Create ad_features as a 1D array
        ad_features = np.array([
            row['City'], row['Region']
        ])
  
        features = np.concatenate([
            ad_features
        ])
        return features, should_bid, paying_price

# File paths
imp_file = r"/media/uas-dtu/OLDUBNT/random/adobe devcraft (1)/ignore/Adobe Devcraft Dataset/dataset/master/imp.txt"
clk_file = r"/media/uas-dtu/OLDUBNT/random/adobe devcraft (1)/ignore/Adobe Devcraft Dataset/dataset/master/clk.txt"
city_file = r"city_embeddings.json"
region_file = r"region_embeddings.json"

# Initialize the dataset
dataset = RTBDataset_new(imp_file, clk_file, city_file, region_file)

# Extract ad_features from the dataset
ad_features_list = []
for i in range(len(dataset)):
    features, _, _ = dataset[i]
    ad_features_list.append(features)  # Append the 1D array of features

# Convert the list of ad_features to a DataFrame
ad_features_df = pd.DataFrame(ad_features_list, columns=[
    'City', 'Region'
])

# Compute the correlation matrix
correlation_matrix = ad_features_df.corr()

# Print the correlation matrix
print(correlation_matrix)

# Plot the correlation heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()