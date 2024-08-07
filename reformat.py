import numpy as np
import pandas as pd

df = pd.read_csv('converted.csv')

df = df.fillna('nan')

print(df.head(5))
df.to_csv('converted.csv', index=False)