mods = ["dotenv", "pandas", "numpy", "streamlit", "psycopg2", "tensorflow", "xgboost"]
for name in mods:
    try:
        __import__(name)
        print(name, "OK")
    except Exception as exc:
        print(name, "ERR", repr(exc))
