import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Constants
REFERENCE_DATE = datetime(1970, 1, 1)

# Helper functions
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

# Load profile and advertiser embeddings
profile_json_path = f"profile.json"
with open(profile_json_path, "r", encoding="utf-8") as f:
    profile_embeddings = json.load(f)

advertiser_json_path = f"avertiser_id.json"
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
            interest_vector[i] = similarity * n_value
    return interest_vector

# RTB Dataset Class
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
            interest_scores = np.zeros(44)
            
        city_embedding = self.city_embeddings_dict.get(str(row['City']), self.city_embeddings_dict["0"])
        region_embedding = self.region_embeddings_dict.get(str(row['Region']), self.region_embeddings_dict["0"])
        ad_features = np.array([
            int(row['Adslotwidth']), int(row['Adslotheight']),
            int(row['Adslotvisibility']), int(row['Adslotformat']),
            float(row['Adslotfloorprice']), row['days_since_epoch'], row['time_part'], row['IP_numeric_normalized']
        ])
  
        features = np.concatenate([
            ad_features,
            city_embedding,
            region_embedding,
            interest_scores
        ])
        return features, should_bid, paying_price

# RL Environment
class RTBEnvironment:
    def __init__(self, dataset, budget, cpa_threshold):
        self.dataset = dataset
        self.current_step = 0
        self.total_steps = len(dataset)
        self.budget = budget
        self.cpa_threshold = cpa_threshold
        self.spent = 0
        self.clicks = 0

    def reset(self):
        self.current_step = 0
        self.spent = 0
        self.clicks = 0
        return self.dataset[0][0]  # Return the initial state

    def step(self, action):
        state, should_bid, paying_price = self.dataset[self.current_step]
        reward = 0
        done = False

        if should_bid and self.spent + paying_price <= self.budget:
            self.spent += paying_price
            self.clicks += 1
            reward = 1  # Reward for a click

            # Penalize if CPA exceeds the threshold
            cpa = self.spent / self.clicks if self.clicks > 0 else 0
            if cpa > self.cpa_threshold:
                reward -= 10  # Heavy penalty for exceeding CPA

        self.current_step += 1
        next_state = self.dataset[self.current_step][0] if self.current_step < self.total_steps else None
        done = self.current_step >= self.total_steps or self.spent >= self.budget

        return next_state, reward, done

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Sample a minibatch from memory
        minibatch = random.sample(self.memory, batch_size)

        # Convert lists of numpy arrays to single numpy arrays
        states = np.array([i[0] for i in minibatch])  # Shape: (batch_size, state_size)
        actions = np.array([i[1] for i in minibatch])  # Shape: (batch_size,)
        rewards = np.array([i[2] for i in minibatch])  # Shape: (batch_size,)
        next_states = np.array([i[3] for i in minibatch])  # Shape: (batch_size, state_size)
        dones = np.array([i[4] for i in minibatch])  # Shape: (batch_size,)

        # Convert numpy arrays to PyTorch tensors
        states = torch.FloatTensor(states)  # Shape: (batch_size, state_size)
        actions = torch.LongTensor(actions).unsqueeze(1)  # Shape: (batch_size, 1)
        rewards = torch.FloatTensor(rewards)  # Shape: (batch_size,)
        next_states = torch.FloatTensor(next_states)  # Shape: (batch_size, state_size)
        dones = torch.FloatTensor(dones)  # Shape: (batch_size,)

        # Compute Q-values for current states
        current_q = self.model(states).gather(1, actions)  # Shape: (batch_size, 1)

        # Compute Q-values for next states
        next_q = self.model(next_states).detach().max(1)[0]  # Shape: (batch_size,)

        # Compute target Q-values
        target = rewards + (1 - dones) * self.gamma * next_q  # Shape: (batch_size,)
        target = target.unsqueeze(1)  # Shape: (batch_size, 1)

        # Compute loss
        loss = nn.MSELoss()(current_q, target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        def load(self, name):
            self.model.load_state_dict(torch.load(name))

        def save(self, name):
            torch.save(self.model.state_dict(), name)

# Training Function
def train_agent(dataset, episodes=1000, batch_size=32):
    env = RTBEnvironment(dataset, budget=10000, cpa_threshold=5)
    state_size = len(dataset[0][0])
    action_size = 2  # Example: bid or not bid
    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(env.total_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    agent.save("rtb_dqn_model.h5")

# Load your dataset
imp_file_path = "../../../../ignore/Adobe Devcraft Dataset/dataset/val/val06.txt"
clk_file_path = "../../../../ignore/Adobe Devcraft Dataset/dataset/clk.06.txt"
city_embeddings_path = "city_embeddings.json"
region_embeddings_path = "region_embeddings.json"

dataset = RTBDataset_new(imp_file_path, clk_file_path, city_embeddings_path, region_embeddings_path)

# Train the agent
train_agent(dataset)