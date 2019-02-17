# Import sas7bdat using sas7bdat 
from sas7bdat import SAS7BDAT
with SAS7BDAT('posts-100.sas7bdat') as sas_file:    
	users_sas_df = sas_file.to_data_frame()
sas_file
dir(sas_file)
sas_file.column_names
sas_file.header
type(users_sas_df)

# Using pandas
import pandas as pd
posts_sas = pd.read_sas('posts-100.sas7bdat')
type(posts_sas)
posts_sas.head()
posts_sas.columns
posts_sas_reader = pd.read_sas('posts-100.sas7bdat', chunksize=10)
posts_sas_reader.read()

