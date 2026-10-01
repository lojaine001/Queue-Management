import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, dbname='iqms', user='postgres', password='0000')
cur = conn.cursor()
cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false")
print("All databases:", cur.fetchall())
cur.execute("SELECT pid, datname, usename, application_name, state, query FROM pg_stat_activity WHERE datname IS NOT NULL")
for row in cur.fetchall():
    print(row)
