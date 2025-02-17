import joblib
import json
import torch
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
import line_profiler

REFERENCE_DATE = datetime(1970, 1, 1)

def process_timestamp(timestamp):
   
    date_part = timestamp[:8]  
    time_part = timestamp[8:] 

    date_obj = datetime.strptime(date_part, "%Y%m%d")
    days_since_epoch = (date_obj - REFERENCE_DATE).days

    return days_since_epoch, int(time_part)


def process_timestamp_optim(timestamp):
    # Parse the entire timestamp with milliseconds
    date_obj = datetime.strptime(timestamp, "%Y%m%d%H%M%S%f")
    
    # Calculate days since the reference date
    days_since_epoch = (date_obj.date() - REFERENCE_DATE.date()).days
    # Extract the time part and convert to integer (HHmmssSSS)
    time_part = int(timestamp[8:])
    
    return days_since_epoch, time_part

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

profile_json_path = f"profile.json"
with open(profile_json_path, "r", encoding="utf-8") as f:
    profile_embeddings = json.load(f)

advertiser_json_path = f"avertiser_id.json"
with open(advertiser_json_path, "r", encoding="utf-8") as f:
    advertiser_embeddings = json.load(f)
    
INTERESTS =[
    "10006", "10024", "10031", "10048", "10052", "10057", "10059", "10063", "10067", "10074",
    "10075", "10076", "10077", "10079", "10083", "10093", "10102", "10684", "11092", "11278",
    "11379", "11423", "11512", "11576", "11632", "11680", "11724", "11944", "13042", "13403",
    "13496", "13678", "13776", "13800", "13866", "13874", "14273", "16593", "16617", "16661",
    "16706", "16751", "10110", "10111"
]
# @profile
def get_interest_scores(profile_interests, advertiser_id):
    # global INTERESTS
    interest_vector = np.zeros(44)  
    advertiser_data = advertiser_embeddings.get(str(advertiser_id), {})
    advertiser_embedding = np.array(advertiser_data.get("embed", [0] * 512))
    n_value = float(advertiser_data.get("N", 0))+1
    # for interest_id in profile_interests.split(','):
    for i, interest_id in enumerate(INTERESTS):
        if interest_id in profile_interests:
            profile_embedding = np.array(profile_embeddings.get(interest_id, {}).get("embed", [0] * 512))
            # similarity = cosine_similarity([advertiser_embedding], [profile_embedding])[0][0]
            # print(" ",similarity)
            similarity = np.dot(advertiser_embedding, profile_embedding) / (np.linalg.norm(advertiser_embedding) * np.linalg.norm(profile_embedding))
            # print(" ",similarity)
            # interest_vector[INTERESTS.index(interest_id)] = similarity*n_value 
            interest_vector[i] = similarity*n_value 

    return interest_vector


with open('city_embeddings.json', 'r') as f:
        city_embeddings_dict = json.load(f)

with open('region_embeddings.json', 'r') as f:
        region_embeddings_dict = json.load(f)
# @profile
def Data_processor(bidRequest):
    global city_embeddings_dict,region_embeddings_dict
    interest_scores=get_interest_scores(bidRequest.user_tags,bidRequest.advertiser_id)
    time=process_timestamp(bidRequest.timestamp)
    ad_features = np.array([
            int(bidRequest.ad_slot_width), int(bidRequest.ad_slot_height),
            int(bidRequest.ad_slot_visibility), int(bidRequest.ad_slot_format),
            float(bidRequest.ad_slot_floor_price),time[0], time[1], ip_to_numeric(bidRequest.ip_address)
        ])

    city_embedding = city_embeddings_dict.get(bidRequest.city, city_embeddings_dict["0"])
    region_embedding = region_embeddings_dict.get(bidRequest.region, region_embeddings_dict["0"])

    features = np.hstack([
            ad_features,
            city_embedding,
            region_embedding,
            interest_scores
        ])
    return features 

