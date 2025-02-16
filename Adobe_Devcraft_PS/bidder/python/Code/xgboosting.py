import json
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier,XGBRegressor
import numpy as np

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

city_map = load_mappings(r"../city.txt")
region_map = load_mappings(r"../region.txt")

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
    def __init__(self, imp_file, clk_file, city_file, region_file, limit=50000):
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
        return self.limit
        # with open(self.imp_file, "r", encoding="utf-8") as f:
        #     return sum(1 for _ in f)

    def __getitem__(self, idx):
        with open(self.imp_file, "r", encoding="utf-8") as f:
            flag=0
            for i, line in enumerate(f):
                if i == idx:
                    values = line.strip().split("\t")
                    if len(values)==24:
                        break
                    elif len(values)
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


if __name__ == "__main__":

    imp_file = r"../../../../ignore/Adobe Devcraft Dataset/dataset/imp.06.txt"
    clk_file = r"../../../../ignore/Adobe Devcraft Dataset/dataset/conv.06.txt"
    city_file = r"../city.txt"
    region_file = r"../region.txt"

    dataset = RTBDataset(imp_file, clk_file, city_file, region_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    all_features = []
    all_should_bid = []
    all_paying_price = []

    for batch in dataloader:
        features, should_bid, paying_price = batch
        all_features.append(features)
        all_should_bid.append(should_bid)
        all_paying_price.append(paying_price)

    X = np.vstack(all_features)
    y_bid = np.hstack(all_should_bid)
    y_price = np.hstack(all_paying_price)


    X_train, X_test, y_bid_train, y_bid_test, y_price_train, y_price_test = train_test_split(
        X, y_bid, y_price, test_size=0.2, random_state=42
    )

  
    xgb_params_bid = {
        "n_estimators": 100, 
        "max_depth": 4,       
        "learning_rate": 0.1,  
        "objective": "binary:logistic", 
        "eval_metric": "logloss",  
        "use_label_encoder": False,
        "subsample": 0.8,    
        "colsample_bytree": 0.8, 
        "reg_alpha": 0.1,    
        "reg_lambda": 0.1,    
        "seed": 42           
    }

    xgb_params_price = {
        "n_estimators": 100,  
        "max_depth": 6,      
        "learning_rate": 0.1, 
        "objective": "reg:squarederror", 
        "eval_metric": "rmse", 
        "subsample": 0.8,    
        "colsample_bytree": 0.8, 
        "reg_alpha": 0.1,     
        "reg_lambda": 0.1,    
        "seed": 42            
    }

   
    model_bid = XGBClassifier(**xgb_params_bid)
    print('started training')
    model_bid.fit(X_train, y_bid_train)

  
    y_bid_pred = model_bid.predict(X_test)
    accuracy = accuracy_score(y_bid_test, y_bid_pred)
    print(f"Bidding Accuracy: {accuracy:.4f}")

    model_price = XGBRegressor(**xgb_params_price)
    model_price.fit(X_train, y_price_train)


    y_price_pred = model_price.predict(X_test)
    rmse = mean_squared_error(y_price_test, y_price_pred)
    print(f"Paying Price RMSE: {rmse:.4f}")