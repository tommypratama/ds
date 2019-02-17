# Harness the full power of SQL
engine_mysql.table_names() 
nicer_query = "SELECT posts.Id, Users.DisplayName, posts.AnswerCount, posts.ViewCount FROM posts INNER JOIN users on posts.OwnerUserId=Users.Id ORDER BY posts.ViewCount DESC LIMIT 5;"
posts = pd.read_sql(nicer_query, engine_mysql) 


