import json
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import pandas as pd
from tqdm import tqdm

# Constants
MODEL_NAME = "sentence-transformers/sentence-t5-xl"
model = SentenceTransformer(MODEL_NAME)

# Load embeddings
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

# Helper functions
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
    try:
        ip_address = replace_x_with_127(ip_address)
        octets = list(map(int, ip_address.split(".")))
        numeric_ip = (octets[0] << 24) + (octets[1] << 16) + (octets[2] << 8) + octets[3]
        return numeric_ip
    except (IndexError, ValueError, AttributeError):
        return 127

def load_click_data(file_path):
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
            interest_vector[i] = similarity * n_value
    return interest_vector

# Dataset class
class RTBDataset(Dataset):
    def __init__(self, imp_file, clk_file, city_file, region_file, limit=10):
        self.imp_file = imp_file
        self.clk_file = clk_file
        self.city_file = city_file
        self.region_file = region_file
        self.limit = limit
        self.city_map = load_mappings(city_file)
        self.region_map = load_mappings(region_file)
        self.clicked_bid_ids = load_click_data(clk_file)
        self.ip_addresses = []
        with open(imp_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                print("feragfwaerfga")
                values = line.strip().split("\t")
                if len(values)<24:
                    print('skipping')
                    continue
                ip_address = values[5]
                numeric_ip = ip_to_numeric(ip_address)
                self.ip_addresses.append(numeric_ip)
        self.min_ip = min(self.ip_addresses)
        self.max_ip = max(self.ip_addresses)

    def __len__(self,limit=10):
        return limit

    def __getitem__(self, idx):
        with open(self.imp_file, "r", encoding="utf-8") as f:
            flag=0
            for i, line in enumerate(f):
                if i == idx:
                    values = line.strip().split("\t")
                    if len(values)==24:
                        break
                elif flag==1 :
                    values = line.strip().split("\t")
                    if len(values)==24:
                        break
                elif i==idx:
                    flag=1
        print("hello in getitem")
        bid_id = values[0]
        should_bid = 1 if bid_id in self.clicked_bid_ids else 0
        paying_price = float(values[20])
        ad_width = int(values[13])
        ad_height = int(values[14])
        ad_format = int(values[16])
        ad_visibility = int(values[15])
        ad_floor_price = float(values[17])
        ip_address = values[5]
        numeric_ip = ip_to_numeric(ip_address)
        normalized_ip = (numeric_ip - self.min_ip) / (self.max_ip - self.min_ip)
        city_name = self.city_map.get(values[7], "unknown")
        region_name = self.region_map.get(values[6], "unknown")
        city_embedding = encode_text([city_name])[0].cpu().numpy()
        region_embedding = encode_text([region_name])[0].cpu().numpy()
        days_since_epoch, time_value = process_timestamp(values[1])
        profile_interests = values[23].split(",")
        advertiser_id = values[22]
        interest_scores = get_interest_scores(profile_interests, advertiser_id)

       
        features = np.concatenate([
            [ad_width, ad_height, ad_format, ad_visibility, ad_floor_price, days_since_epoch, time_value, normalized_ip],
            city_embedding,
            region_embedding,
            interest_scores
        ])

        return features, should_bid, paying_price
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
        region_embedding = self.region_embeddings_dict.get(str(row['Region']), self.region_embeddings_dict["0"])
        ad_features = np.array([
            int(row['Adslotwidth']), int(row['Adslotheight']),
            int(row['Adslotvisibility']), int(row['Adslotformat']),
            float(row['Adslotfloorprice']),row['days_since_epoch'], row['time_part'], row['IP_numeric_normalized']
        ])
  
       
        features = np.concatenate([
            ad_features,
            city_embedding,
            region_embedding,
            interest_scores
        ])
        return features , should_bid, paying_price
    
def plot_tsne(features, labels, title="t-SNE Visualization"):
    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label="Should Bid (0 or 1)")
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

# Main script
if __name__ == "__main__":
    # Initialize dataset and dataloader
    dataset = RTBDataset_new(
         r"/media/uas-dtu/OLDUBNT/random/adobe devcraft (1)/ignore/Adobe Devcraft Dataset/dataset/master/imp.txt",
        r"/media/uas-dtu/OLDUBNT/random/adobe devcraft (1)/ignore/Adobe Devcraft Dataset/dataset/master/clk.txt",
         r"city_embeddings.json",
         r"region_embeddings.json"
         )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Collect features and labels
    features_list = []
    labels_list = []
    for features, should_bid, _ in tqdm(dataloader,desc='loading data'):
        features_list.append(features.numpy())
        labels_list.append(should_bid.numpy())

    # Concatenate all features and labels
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    # Plot t-SNE
    plot_tsne(all_features, all_labels, title="t-SNE Visualization of RTB Dataset")