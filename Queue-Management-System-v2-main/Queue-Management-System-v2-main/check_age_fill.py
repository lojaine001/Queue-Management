import psycopg2
conn = psycopg2.connect(host="localhost", port=5432, dbname="iqms", user="postgres", password="0000")
cur = conn.cursor()
cur.execute("""
    SELECT
      COUNT(*) AS total_eligible,
      COUNT(age_estimate) AS with_age,
      ROUND(100.0 * COUNT(age_estimate) / NULLIF(COUNT(*),0), 1) AS pct_age,
      COUNT(*) FILTER (WHERE gender IS NOT NULL AND gender != 'unknown') AS with_gender,
      ROUND(100.0 * COUNT(*) FILTER (WHERE gender IS NOT NULL AND gender != 'unknown') / NULLIF(COUNT(*),0), 1) AS pct_gender
    FROM entrance_events
    WHERE camera_id NOT LIKE 'SIM_%'
      AND dwell_seconds >= 10
""")
print("OVERALL (total, with_age, pct_age, with_gender, pct_gender):", cur.fetchone())
cur.close(); conn.close()
