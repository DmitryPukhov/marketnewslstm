import pandas as pd
import numpy as np

cols_agg = {'price': ['min', 'mean', 'max', 'count']}

df = pd.DataFrame([['asset1', '2018-11-17', 10],
                   ['asset1', '2018-11-17', 11],
                   ['asset1', '2018-11-18', 12],
                   ['asset1', '2018-11-29', 13],
                   ['asset2', '2018-11-18', 14],
                   ['asset2', '2018-11-29', 15]],
                  columns=['assetCode', 'time', 'price'], index=[10, 11, 12, 13, 14, 15])
df.time = df.time.astype('datetime64')

print(df)

df = df.groupby(['assetCode', 'time']).mean().reset_index(['assetCode', 'time'])
print(df)

rg = df.groupby('assetCode').rolling('2D', on='time').apply(np.mean).reset_index('assetCode')
rgagg=df.groupby('assetCode').rolling('2D', on='time').agg(cols_agg)

print(rg)
