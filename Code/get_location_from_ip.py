import pandas as pd

def get_location_from_ip(ip:str,database_csv_path:str)->dict:
    df=pd.read_csv(database_csv_path,index_col=['start_ip','end_ip','country code','country','state','city'],index_col=False,low_memory=False)
    print('the lookup table is loaded')
    ip=[int(val.strip()) for val in ip.split('.')[:-1]]
    assert ip.split('.')[-1]=='*','the format of ip is not right'
    possible_ip_range=[256**3*ip[0]+256**2*ip[1]+256**1*ip[2]+0,256**3*ip[0]+256**2*ip[1]+256**1*ip[2]+255]
    
    
    
    
    