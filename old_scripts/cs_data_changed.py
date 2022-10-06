import pandas as pd
import os
from data_factory import DataFactory
from pathlib import Path
data = 'cs'
datapath=os.path.join('data','CS')
data_factory = DataFactory()

header_l = ["text"]
new_data = pd.read_csv("data/CS/cs.csv", on_bad_lines='skip')
new_data.to_csv("cs.csv",header=header_l, index=False)