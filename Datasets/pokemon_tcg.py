import json, pandas as pd, numpy as np
import json

def cardinality(ip):
    if(ip == ['']):
        return 0
    return len(ip)*2

df = pd.read_excel(r'trading-cards.xlsx',engine='openpyxl')
df = df[["skill_name","skill_cost","skill_damage","damage_mult","skill_text"]]
df['damage_mult'].replace('',np.nan,inplace=True)
df.dropna(subset=['skill_cost'], inplace=True)
for index, row in df.iterrows():
    line = row['skill_cost'].strip('][').split(',')
    df.at[index,'skill_cost']=cardinality(line)


test_power = df[df['skill_damage'].isna()]
test_desc = df[df['skill_text'].isna()]
full_data = df.dropna(subset=["skill_text",'skill_damage']) #(df-test_power-test_desc)
df.dropna(subset=['skill_damage'],inplace = True)
df.reset_index(drop=True, inplace=True)
test_power.reset_index(drop=True, inplace=True)
test_desc.reset_index(drop=True, inplace=True)
print(df["damage_mult"].isna().sum())
print(df)#includes test_des
print(test_power)
print(test_desc)
