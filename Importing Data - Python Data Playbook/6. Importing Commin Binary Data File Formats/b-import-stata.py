# Import stata with pandas
import pandas as pd
posts_stata = pd.read_stata('posts-100.dta')
type(posts_stata)
dir(posts_stata)
posts_stata.columns
posts_stata.head()
