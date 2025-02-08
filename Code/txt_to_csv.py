import pandas as pd
import sys 
import os
from multiprocessing import Pool

def process_chunk(chunk):
    # Process the chunk (e.g., filter, transform)
        return chunk
def save_csv(file:str)->pd.DataFrame:

    # chunks = pd.read_csv("your_file.csv", chunksize=100000)
    with open(file,'r') as f:
        first_line=f.readline()
        # print(first_line)
        num_columns=len(first_line.split('\t'))
        if 'bid' in file:
            assert num_columns==21,num_columns
            col_names=['BidID','Timestamp','VisitorID','User-Agent','IP','Region','City','Adexchange','Domain','URL','AnonymousURLID','AdslotID','Adslotwidth','Adslotheight','Adslotvisibility','Adslotformat','Adslotfloorprice','CreativeID','Biddingprice','AdvertiserID','profile']
        else:
            assert num_columns==24,num_columns
            col_names=['BidID','Timestamp','Logtype','VisitorID','User-Agent','IP','Region','City','Adexchange','Domain','URL','AnonymousURLID','AdslotID','Adslotwidth','Adslotheight','Adslotvisibility','Adslotformat','Adslotfloorprice','CreativeID','Biddingprice','Payingprice','KeypageURL','AdvertiserID','profile']

        # print(num_columns)
    df = pd.read_csv(file,names=col_names,index_col=False, delimiter="\t",low_memory=False,chunksize=100000)
    with Pool(processes=4) as pool:
        results = pool.map(process_chunk, df)
    results=pd.concat(results)
    return results


    # df.to_csv(save_dir+os.sep+file.replace('txt','csv'), index=False)


# if __name__=='__main__':
#     num_processor=multiprocessing.cpu_count()

#     with multiprocessing.Pool(processes=int(num_processor*0.7)) as pool:
#         pool.map(save_csv,os.listdir(dir_path))