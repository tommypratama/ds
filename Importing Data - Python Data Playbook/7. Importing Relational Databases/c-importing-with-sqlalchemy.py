# Import data using SQL Alchemy
from sqlalchemy import create_engine
engine = create_engine('sqlite:///importing_sqlite.db')
type(engine)
dir(engine)
engine.table_names()
engine.url
engine.dialect
engine.driver

# Connect to your database of choice
engine_sqlite = create_engine('sqlite:///importing_sqlite.db')
engine_mysql = create_engine('postgresql://xavier:postgres@localhost:5432/importing_postgres')
engine_postgresql = create_engine('mysql+mysqlconnector://root:mysql@localhost:3306/importing_mysql')

# And you get an engine, which you can use 
engine_sqlite = create_engine('sqlite:///importing_sqlite.db')
engine_postgres = create_engine('postgresql://xavier:postgres@localhost:5432/importing_postgres')
engine_mysql = create_engine('mysql+mysqlconnector://root:mysql@localhost:3306/importing_mysql')
