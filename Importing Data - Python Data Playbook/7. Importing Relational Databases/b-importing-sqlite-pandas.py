# Import data from SQLite using pandas
import pandas as pd 
stack_connection = sqlite3.connect('importing_sqlite.db') 
posts_df = pd.read_sql("select * from posts;", stack_connection) 
type(posts_df)
posts_df.columns
posts_df.head()

