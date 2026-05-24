import psycopg2
c = psycopg2.connect(host="localhost",dbname="iqms",user="postgres",password="0000").cursor()
c.execute("SELECT COUNT(*) FROM entrance_events WHERE timestamp BETWEEN '2026-05-20 17:21:00' AND '2026-05-20 17:22:00'")
print("People:", c.fetchone()[0])
