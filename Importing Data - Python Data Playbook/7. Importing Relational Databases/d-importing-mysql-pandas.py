# Import data from MySQL with pandas
# show databases;
# use importing_mysql
# show tables;
engine = create_engine('mysql+mysqlconnector://root:mysql@localhost:3306/importing_mysql')
posts = pd.read_sql_table('posts', engine, index_col='Id')
type(posts)
posts.columns 
posts.head()

# Several parameters available
posts = pd.read_sql_table('posts', engine, columns=['Id', 'CreationDate', 'Tags'])
posts.head()
type(posts.iloc(1)[1])
posts = pd.read_sql_table('posts', engine, columns=['Id', 'CreationDate', 'Tags'], parse_dates={'CreationDate': {'format': '%Y-%m-%dT%H:%M:%S.%f'}})
type(posts.iloc(1)[1])
