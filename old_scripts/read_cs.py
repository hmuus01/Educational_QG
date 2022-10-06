import pandas as pd

# df = pd.read_csv('data/cs.csv', sep='\t', header=0)
df = pd.read_csv('data/cs2.csv', sep=',', header=0)
df = df.reset_index()
df=df[['abstractText']]
df.to_csv('data//cs/cs.csv', index=False)
print()
