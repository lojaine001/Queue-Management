import re
code = open("utils/db_logger.py").read()
method = """
    def get_today_entry_timestamps(self):
        if not self.enabled:
            return []
        try:
            self.cursor.execute("""
                SELECT EXTRACT(EPOCH FROM timestamp)
                FROM entrance_events
                WHERE timestamp >= CURRENT_DATE
                ORDER BY timestamp ASC
            """)
            return [float(row[0]) for row in self.cursor.fetchall()]
        except Exception as e:
            print(f"[DB] get_today_entry_timestamps error: {e}")
            return []

"""
code = code.replace("    def close(self):", method + "    def close(self):")
open("utils/db_logger.py", "w").write(code)
print("Done")
