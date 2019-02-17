# Import data with Psycopg2
# psql
# \l
# \c importing_postgres
# \d
# SELECT "Id", "Title" FROM posts LIMIT 5;
import psycopg2
stack_connection = psycopg2.connect("dbname=importing_postgres user=xavier host=localhost")
so_cursor = stack_connection.cursor()

# Execute the query and get the results
so_cursor.execute("select * from posts")
first_row = so_cursor.fetchone()
first_row
type(first_row)
rows = so_cursor.fetchall()
rows
type(rows)

# Don't forget to commit and close connection
stack_connection.commit()
stack_connection.close()
