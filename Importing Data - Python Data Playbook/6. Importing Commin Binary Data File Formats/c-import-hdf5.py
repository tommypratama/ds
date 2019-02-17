# Import hdf5 using h5py
import h5py
file = h5py.File("posts-100.h5",'r')
dataset = file['posts']
for x in dataset['table']:
    print(x)

# Using pandas
import pandas as pd
posts_hdf = pd.read_hdf('posts-100.h5', 'posts')
posts_hdf.columns
posts_hdf.keys()
pd.read_hdf('posts-100.h5', 'posts', start=2, stop=5, columns=['CreationDate','Title','Tags']).head()
pd.read_hdf('posts-100.h5', 'posts', columns=['Score', 'Tags'], where='Score>10 or Tags = "<machine-learning>"').head()
