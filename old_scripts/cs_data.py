import pandas as pd
import os

from tqdm import tqdm

from data_factory import DataFactory
from pathlib import Path
data = 'cs'
datapath=os.path.join('data','CS')
data_factory = DataFactory()


cs_df = data_factory.get_dev(data)
df_out = pd.DataFrame(['text'])
# print(cs_df.head())
for i, row in tqdm(cs_df.iterrows()):
    dict_item=row.item()
    if len(dict_item) < 1:
        continue
    text = dict_item[0]['text']
    entry = pd.DataFrame.from_dict({
        "text": [text]
    })

    df_out = pd.concat([df_out, entry], ignore_index=True)
df_out.to_csv(os.path.join(datapath, 'cs2.csv'), sep='\t',header=False, index=False)


# new_data = pd.read_csv("data/CS/cs.csv", on_bad_lines='skip')
# new_data.to_csv("cs.csv",sep='\t')
# text = [li['text'] for li in cs_df]

# text = [d['text'] for d in cs_df]
# print(text)

#
# filepath = Path('QG/data/cs.csv')
#
# filepath.parent.mkdir(parents=True, exist_ok=True)
#
# cs_df.to_csv("cs.csv", sep='\t')

# new_cs_df = pd.read_csv('data/cs_small/cs.csv', error_bad_lines=False)

# new_cs_df.to_csv("cs.csv")