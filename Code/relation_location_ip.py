import pandas as pd 
from multiprocessing import Pool
import matplotlib.pyplot as plt
from txt_to_csv import save_csv
from tqdm import tqdm

csv_file='/home/akshit/adobe devcraft/Adobe Devcraft Dataset/dataset/bid.07.txt'
features=['City','Region']
def ip_to_numericip(ip:str)->int:
    # print(ip)
    if pd.isna(ip):
        return 0
    ip=str(ip)
    assert len(ip.split("."))==4 and ip.split('.')[-1]=='*',f'the ip is not in correct form ,{type(ip)} {ip}'
    nums=[int(val) for val in ip.split(".")[:-1]]
    ip=[nums[-(i+1)]*(256)**i for i in range(len(nums))]
    return sum(ip)+127

df=save_csv(csv_file)

IPs=df['IP'].tolist()
with Pool(processes=4) as pool:
    results = pool.map(ip_to_numericip, IPs)

results=pd.Series(results)
for feature in features:
    print(f'displaying plots for the feature:{feature}')
    unique=df[feature].unique()
    print('number of unique values :',len(unique))
    print(sorted(unique))
    colors=['red','blue','green','yellow','black']
    count=0
    for val in unique:
        indexes = df.loc[df[feature] == val].index
        plt.hist(results[indexes], density=True,bins=50, edgecolor="black", alpha=0.5,color=colors[count],label=val)
        count+=1
        count%=5
        if count==4:
            plt.title(f'histogram for feature:{feature}')
            plt.legend(title="Categories")
            # plt.show()
            
    plt.show()
