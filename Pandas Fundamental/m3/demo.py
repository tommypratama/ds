# -*- coding: utf-8 -*-
import pandas as pd
import os

# Let's load the data for the first time
df = pd.read_pickle(os.path.join('..', 'data_frame.pickle'))

# Demo 1
df.artist
artists = df['artist']
pd.unique(artists)
len(pd.unique(artists))

# Demo 2
s = df['artist'] == 'Bacon, Francis'
s.value_counts()
 
# Other way
artist_counts = df['artist'].value_counts()
artist_counts['Bacon, Francis']

# Demo 3
df.loc[1035, 'artist']
df.iloc[0, 0]
df.iloc[0, :]
df.iloc[0:2, 0:2]

# Try multiplication
df['height'] * df['width']
df['width'].sort_values().head()
df['width'].sort_values().tail()

# Try to convert
pd.to_numeric(df['width'])

# Force NaNs 
pd.to_numeric(df['width'], errors='coerce')
df.loc[:, 'width'] = pd.to_numeric(df['width'], errors='coerce')

pd.to_numeric(df['height'], errors='coerce')
df.loc[:, 'height'] = pd.to_numeric(df['height'],
                                    errors='coerce')

df['height'] * df['width']
df['units'].value_counts()

# Assign - create new columns with size
area = df['height'] * df['width']
df = df.assign(area=area)

df['area'].max()
df['area'].idxmax()
df.loc[df['area'].idxmax(), :]
