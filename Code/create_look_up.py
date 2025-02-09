import json
from get_sentence_piece_embed import encode_text
from tqdm import tqdm

file_path='/home/akshit/adobe devcraft/Adobe Devcraft PS/user.profile.tags.txt'
filename='profile.json'
with open(file_path,'r') as f:
    data=f.readlines()
print(*[words.split('\t') for words in data])
look_up={}
for d in tqdm(data):
    val={'text':d[1].strip(),'embed':encode_text(d[1].strip())}
    look_up[d[0]]=val
with open(filename, "w", encoding="utf-8") as f:
    json.dump(look_up, f, indent=4, ensure_ascii=False)
print(f"JSON saved to {filename}")