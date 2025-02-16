import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/sentence-t5-xl"
model = SentenceTransformer(MODEL_NAME)

with open("profile.json", "r", encoding="utf-8") as f:
    profile_embeddings = json.load(f)

with open("avertiser_id.json", "r", encoding="utf-8") as f:
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

city_map = load_mappings(r"../Adobe Devcraft PS/city.txt")
region_map = load_mappings(r"../Adobe Devcraft PS/region.txt")

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

def min_max_normalize(values, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in values]

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

class RTBDataset(Dataset):
    def __init__(self, imp_file, clk_file, city_file, region_file, limit=100):
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
                values = line.strip().split("\t")
                if len(values)<24:
                    print('skipping')
                    continue
                ip_address = values[5]
                numeric_ip = ip_to_numeric(ip_address)
                self.ip_addresses.append(numeric_ip)
        self.min_ip = min(self.ip_addresses)
        self.max_ip = max(self.ip_addresses)

    def __len__(self):
        with open(self.imp_file, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

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
        city_embedding = encode_text([city_name])[0]
        region_embedding = encode_text([region_name])[0]
        days_since_epoch, time_value = process_timestamp(values[1])
        profile_interests = values[23].split(",")
        advertiser_id = values[22]
        interest_scores = get_interest_scores(profile_interests, advertiser_id)
        num_features = torch.tensor([
            ad_width, ad_height, ad_format, ad_visibility, ad_floor_price,
            days_since_epoch, time_value, normalized_ip
        ], dtype=torch.float)
        should_bid = torch.tensor(should_bid, dtype=torch.float)
        paying_price = torch.tensor(paying_price, dtype=torch.float)
        interest_scores = torch.tensor(interest_scores, dtype=torch.float)
        return num_features, city_embedding, region_embedding, interest_scores, torch.tensor([should_bid, paying_price])

if __name__ == "__main__":
    imp_file = r"../Adobe Devcraft Dataset/dataset/imp.06.txt"
    clk_file = r"../Adobe Devcraft Dataset/dataset/clk.06.txt"
    city_file = r"../Adobe Devcraft PS/city.txt"
    region_file = r"../Adobe Devcraft PS/region.txt"

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
        break