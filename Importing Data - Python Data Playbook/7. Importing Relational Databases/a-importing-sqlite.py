# Import data from SQLite using sqlite3
import sqlite3
# ls
stack_connection = sqlite3.connect('importing_sqlite.db')
type(stack_connection)
stack_cursor = stack_connection.cursor()
stack_cursor.execute("select name from sqlite_master where type = 'table';")
stack_cursor.fetchone()

# Careful on names
# ls -l 
stack_connection_bad = sqlite3.connect('bad_name_sqlite.db')
stack_connection_bad.cursor().execute("select name from sqlite_master where type = 'table';").fetchall()
# ls -l

# Query your data
rows = stack_cursor.execute('select * from posts').fetchall()
type(rows)
rows[0]
type(rows[0])
stack_cursor.execute('select * from posts limit 1').fetchall()
stack_cursor.execute('select Id, Score, Tags from posts limit 3').fetchall()