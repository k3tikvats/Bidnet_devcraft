import matplotlib.pyplot as plt
import os
from txt_to_csv import save_csv
import seaborn as sns
features=['Region','City','Adslotwidth','Adslotheight','Adslotfloorprice','Biddingprice','Payingprice']
features_not_in_bid=['Payingprice']
csv_dir='/home/akshit/adobe devcraft/Adobe Devcraft Dataset/dataset'
for file in os.listdir(csv_dir):
    file_path=csv_dir+os.sep+file
    df=save_csv(file_path)
    print(df.columns)
    for feature in features:
        if 'bid' in file and feature in features_not_in_bid:
            continue
        plt.subplot(2,1,1)
        plt.hist(df[feature].dropna(), bins=30, edgecolor="black", alpha=0.7)
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {feature}")
        # plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.subplot(2,1,2)
        sns.boxplot(x=df[feature])
        plt.title(f'Box Plot of {feature}')
        plt.show()


    