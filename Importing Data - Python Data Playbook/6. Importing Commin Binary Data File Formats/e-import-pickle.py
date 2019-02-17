# Import pickle files using the pickle library
import pickle
with open('posts-100.pkl.gz', 'rb') as pickle_file:
    posts_pickle = pickle.load(pickle_file)
type(posts_pickle)
posts_pickle.columns
posts_pickle.head()

# Import using pandas
import pandas as pd
posts_pickle = pd.read_pickle('posts-100.pkl')
type(posts_pickle)
posts_pickle.columns
posts_pickle.head()

