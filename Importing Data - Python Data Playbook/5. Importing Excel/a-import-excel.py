import pandas as pd

# Create ExcelFile
excel_file = pd.ExcelFile('stackoverflow.xlsx')
type(excel_file)
excel_file.sheet_names

# Parse into DataFrame
excel_df = excel_file.parse()
type(excel_df)
excel_df.head()
excel_df

# Or use pandas directly to load worksheet, with read_excel
posts_excel = pd.read_excel('stackoverflow-one.xlsx')
type(posts_excel)
dir(posts_excel)
posts_excel.columns
posts_excel.head()
pd.read_excel('stackoverflow-one.xlsx', usecols=[0, 3]).columns
pd.read_excel('stackoverflow-one.xlsx', usecols='A:C').columns
pd.read_excel('stackoverflow-one.xlsx', usecols='A,C').columns

# Get a dict of worksheets
excel_file.sheet_names
posts_dict = pd.read_excel('stackoverflow.xlsx',sheet_name=None)
type(posts_dict)
posts_dict.keys()
posts_dict['Posts'].head()

# Different ways of getting the data you need
posts_dict['Users'].head()
pd.read_excel('stackoverflow.xlsx',sheet_name='Users').head()
pd.read_excel('stackoverflow.xlsx',sheet_name='Users', usecols=range(1,9)).head()
pd.read_excel('stackoverflow.xlsx',sheet_name=2).head()
pd.read_excel('stackoverflow.xlsx',sheet_name='Users', usecols=range(1,9)).head()
pd.read_excel('stackoverflow.xlsx',sheet_name='Users', usecols=range(1,9),skiprows=4).head()
pd.read_excel('stackoverflow.xlsx',sheet_name='Users', usecols=range(1,9),nrows=2).head()
pd.read_excel('stackoverflow.xlsx',sheet_name='Users', usecols=range(1,9)).dtypes
pd.read_excel('stackoverflow.xlsx',sheet_name='Users', usecols=range(1,9), dtype={'PostTypeId': str}).dtypes
pd.read_excel('stackoverflow.xlsx',sheet_name='Users', converters={'Id': lambda x: x + 1000}).head()
pd.read_excel('stackoverflow.xlsx',sheet_name='Posts', usecols=[0,7,8]).head()
pd.read_excel('stackoverflow.xlsx',sheet_name='Posts', usecols=[0,7,8], keep_default_na=False).head()



