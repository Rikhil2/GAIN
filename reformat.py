import pandas as pd

df = pd.read_csv('data/converted.csv')

df = df.fillna('nan')

print(df.head(5))
df.to_csv('data/converted.csv', index=False)