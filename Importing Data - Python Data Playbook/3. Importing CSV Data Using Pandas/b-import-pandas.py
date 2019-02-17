import pandas as pd

# Import data using pandas with read_csv
posts_csv = pd.read_csv('posts-100.csv')
type(posts_csv)
posts_csv
posts_csv.head()
posts_csv.head(3)

# You can get the numpy array from the DataFrame
posts_csv.values()
type(posts_csv.values())

# Import from a URL
remote_file = 'https://raw.githubusercontent.com/xmorera/sample-data/master/csv/posts-100.csv'
posts_url = pd.read_csv(remote_file, header=None)
posts_url.head()

# You can read a small number of lines too
posts_small = pd.read_csv('posts-100.csv', nrows=3)
posts_small
posts_small = pd.read_csv('posts-100.csv', nrows=3, skiprows=3)
posts_small

# Use a lambda to specify which rows to skip
posts_odd = pd.read_csv('posts-100.csv', skiprows=lambda x: x % 2 != 0)
posts_odd.head()

# You can also specify that you want to load only certain columns
posts_columns = pd.read_csv('posts-100.csv', usecols=[0,6,7,8])
posts_columns.head(5)

# The DataFrame gives a name to columns, but this does not look right
posts_columns.columns

# So we specify that the file does not have a header, and now labels are added automatically
posts_no_header = pd.read_csv('posts-100.csv', header=None)
posts_no_header.columns

# And you can add a prefix for column names when no header info exists
posts_prefix = pd.read_csv('posts-100.csv', header=None, prefix='Col')
posts_prefix.columns

# And you can add column names
header_fields = ['New_Id', 'New_PostTypeId', 'New_CreationDate', 'New_Score', 'New_ViewCount', 'New_LastActivityDate', 'New_Title', 'New_Tags', 'New_AnswerCount', 'New_CommentCount', 'New_FavoriteCount', 'New_ClosedDate']
posts_add_header = pd.read_csv('posts-100.csv', names=header_fields)

# Even easier if headers are included in the file
posts_header = pd.read_csv('posts-100-header.csv') 
posts_header.columns

# Headers are important because you can use them to refer to columns
posts_header['answer_count'].head()
posts_header[['Id','PostTypeId']].head()

# Although you might want to actually remove the headers
pd.read_csv('posts-100-header.csv', skiprows=1).head()

# Specify types
pd.read_csv('posts-100-header.csv', usecols=[0, 1, 2, 7]).head()
pd.read_csv('posts-100-header.csv', usecols=[0, 1, 2, 7]).columns()
pd.read_csv('posts-100-header.csv', usecols=[0, 1, 2, 7]).dtypes
pd.read_csv('posts-100-header.csv', usecols=[0, 1, 2, 7], dtype={'PostTypeId': str}).dtypes
pd.read_csv('posts-100-header.csv', usecols=[0, 1, 2, 7], dtype={'PostTypeId': float}).dtypes

# Can also use a converter on a column
posts_date.head()
posts_date.iloc[1]
posts_date.iloc[1]['Tags']
type(posts_date.iloc[1]['Tags'])

# Apply a function with a regular expression
posts_tags = pd.read_csv('posts-100-header.csv', usecols=[0, 1, 2, 7], converters={'Tags': lambda x: re.findall('<[A-Za-z0-9_-]*>',x)})

# Work with dates too
type(pd.read_csv('posts-100-header.csv', usecols=[0, 1, 2, 7])['CreationDate'][0])
posts_date = pd.read_csv('posts-100-header.csv', usecols=[0, 1, 2, 7], parse_dates=['CreationDate'])
type(posts_date['CreationDate'][0])

# Let's see some missing values
posts_missing = pd.read_csv('posts-100-header.csv', usecols=[0, 3, 4, 8, 9, 10])

# Work with missing values
pd.read_csv('posts-100-header.csv', usecols=[0, 3, 4, 8, 9, 10], na_filter=False).head()
pd.read_csv('posts-100-header.csv', usecols=[0, 3, 4, 8, 9, 10], na_filter=True).head()

# Now with a tsv file, there is an error unless you set sep or delimiter
posts_tsv = pd.read_csv('posts-100.tsv')
posts_tsv = pd.read_csv('posts-100.tsv', sep='\t')
posts_tsv.head()
posts_tsv = pd.read_csv('posts-100.tsv', delimiter='\t')
posts_tsv.head()
posts_tsv =  pd.read_table('posts-100.tsv')
posts_tsv.head()




























