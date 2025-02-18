import joblib
import json
import torch.nn as nn
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
import random
# import line_profiler

REFERENCE_YEAR = 1970
REFERENCE_MONTH = 1
REFERENCE_DAY = 1

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def days_in_month(year, month):
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if month == 2 and is_leap_year(year):
        return 29
    return month_days[month - 1]

def days_since_epoch(year, month, day):
    days = 0

    # Add days from complete years since 1970
    for y in range(REFERENCE_YEAR, year):
        days += 366 if is_leap_year(y) else 365

    # Add days for completed months in the current year
    for m in range(1, month):
        days += days_in_month(year, m)

    # Add remaining days in the current month
    days += day - 1  # Since the day starts from 1

    return days


def stochastic():
    return random.uniform(0,1)
# @profile
def process_timestamp(timestamp):
    date_part = timestamp[:8]
    time_part = timestamp[8:]

    year = int(date_part[:4])
    month = int(date_part[4:6])
    day = int(date_part[6:])

    days_since = days_since_epoch(year, month, day)

    return days_since, int(time_part)


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
    return features,bidRequest.visitor_id,bidRequest.user_agent

class XGBoost(nn.Module):
    def __init__(self,path):
        self.model=joblib.load(path)
    def predict(X):
        if X[-1]=='1' and stochastic()<0.9:
            if stochastic()<0.8:
                return int(X[-2])+stochastic()*50
            else:
                return max(int(X[-2])-stochastic()*100,0)
        else:
            return -1